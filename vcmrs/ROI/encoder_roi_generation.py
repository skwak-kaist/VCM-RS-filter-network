# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import os
import atexit
import cv2
import shutil
import torch
import shutil
import vcmrs
from vcmrs.ROI.roi_generator import roi_accumulation as roi_accumulation
from vcmrs.ROI.roi_syntax import roi_bitenc
import subprocess
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils import datasets as datasets
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.utils import *
from vcmrs.Utils.component import Component
from vcmrs.Utils import io_utils
from vcmrs.Utils import data_utils
from vcmrs.Utils import utils
from vcmrs.Utils.io_utils import makedirs

USE_NATIVE_COLORSPACE = True

# component base class
class roi_generation(Component):
  def __init__(self, ctx):
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.ctx = ctx
    self.cfg = os.path.join(cur_dir_path,"./roi_generator/Towards_Realtime_MOT/cfg/yolov3_1088x608.cfg")
    self.weights = os.path.join(cur_dir_path, "./roi_generator/jde.1088x608.uncertainty.pt")
    self.iou_thres = 0.5
    self.conf_thres = 0.5
    self.nms_thres = 0.4
    self.accumulation_period = 64
    self.roi_update_period = 1
    self.desired_max_obj_size = 200
    self.max_num_rois = 11
    self.track_buffer = 30
    self.img_size = [0, 0]
    self.working_dir = ''
    atexit.register(self.cleanup)
    
  def cleanup(self):
    if self.working_dir != '':
      yuv_temp = os.path.abspath(os.path.join(self.working_dir,"yuv_temp"))
      temp = os.path.abspath(os.path.join(self.working_dir,"temp"))
      if os.path.exists(yuv_temp):
        shutil.rmtree(yuv_temp)
      if os.path.exists(temp):
        shutil.rmtree(temp)
    
    
  def _is_image(self, file_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.svg', '.tiff', '.ico']
    ext = os.path.splitext(file_path)[1]
    return ext.lower() in img_extensions
  
  # generate_RoI_YOLOv4 moved to unified networkDetectRoI function

  # generate_RoI_FasterRCNN moved to unified networkDetectRoI function

  def get_roi_extension_percent( self, qp ):
    if qp >= 22: return 0.0
    a0, a1, a2 = 38.3, -3.5, 0.08
    val = a0 + qp*a1 + qp**2*a2
    val = int( val * 2 + 0.5 ) / 2 ## rounding to the nearest half
    val /= 100       ## conversion to percents
    val = max(0.0, min(1.0, val))
    return val

  def generateRoI(self, input_dir_rgb, input_dir_yuv, bit_depth, output_name_template, item):
    # specify the path to the folder containing .png files
    temp_output_path = io_utils.create_temp_folder_suffix( os.path.abspath(os.path.join(item.args.working_dir,"yuv_temp")) )

    # get a list of all files in the folders
    png_list_rgb = data_utils.get_image_file_list(input_dir_rgb)
    png_list_yuv = data_utils.get_image_file_list(input_dir_yuv)
    
    len_png_list = max( len(png_list_rgb), len(png_list_yuv) )

    input_frame_skip = item.args.FrameSkip # using frame_skip only on the input side.
    frames_to_be_encoded = item.args.FramesToBeEncoded
    seqLength = frames_to_be_encoded
    if seqLength==0:
      seqLength = len_png_list - input_frame_skip

    generator = roi_accumulation.RoIImageGenerator(
      opt=self,
      item=item,
      seqLength=seqLength,
    )

    for i in range(len_png_list):
      if i >= item.args.FrameSkip and i < (input_frame_skip + seqLength):
        
        # load as 8-bit image for descriptor generation
        img_rgb = cv2.imread(os.path.join(os.path.abspath(input_dir_rgb), png_list_rgb[i])) if input_dir_rgb else None

        if input_dir_yuv:
          img_process = cv2.imread( os.path.join(os.path.abspath(input_dir_yuv), png_list_yuv[i]), cv2.IMREAD_UNCHANGED )
        else:
          if bit_depth==8:
            img_process = img_rgb
          else:
            img_process = cv2.imread(os.path.join(os.path.abspath(input_dir_rgb), png_list_rgb[i]), cv2.IMREAD_UNCHANGED) if input_dir_rgb else None

        if len(img_process.shape)>2: # remove 4-th component if exists in the file
          if img_process.shape[2]>3:
            img_process = img_process[:,:,0:3]

        org_size_y, org_size_x, _ = img_process.shape

        # possible different resolutions on each GOP
        update_size_x, update_size_y, rtg_size_x, rtg_size_y, retargeting_gops_rois, retargeting_gops_resolutions = generator.generateRoIImage(self.roi_update_period, self.accumulation_period, self.desired_max_obj_size, self.max_num_rois, img_rgb, img_process, bit_depth, output_name_template)
        
        # if retargettig is enabled this is returned as not None
        if retargeting_gops_rois is not None:
          gops_roi_bytes = roi_bitenc.encode_gops_rois_params(self.roi_update_period, org_size_x, org_size_y, rtg_size_x, rtg_size_y, retargeting_gops_rois, retargeting_gops_resolutions, vcmrs.log)
          item.add_parameter('ROI', param_data=bytearray(gops_roi_bytes) ) 
        
        # update inner codec resolution conly if padding is disabled  
        if update_size_x is not None:
          item.args.SourceWidth = update_size_x
        if update_size_y is not None:
          item.args.SourceHeight = update_size_y

    if item.args.SpatialResamplingFlag :
      self.update_spatial_parameter(item, item.args.SourceWidth, item.args.SourceHeight)
      
    item.OriginalFrameIndices = item.OriginalFrameIndices[item.args.FrameSkip:]
    item.args.FrameSkip = 0 # set current FrameSkip to 0, as on the output it is not used
    
  def process(self, input_fname, output_fname, item, ctx):
    self.working_dir = os.path.abspath(item.args.working_dir)
    
    self.accumulation_period, self.roi_update_period =  get_roi_accumulation_periods(item.args, item.IntraPeriod, item.FrameRateRelativeVsInput)

    self.desired_max_obj_size = 200
    if item.args.RoIGenerationNetwork=="faster_rcnn_X_101_32x8d_FPN_3x": self.desired_max_obj_size = 100
    if item.args.RoIGenerationNetwork=="yolov3_1088x608": self.desired_max_obj_size = 320 

    self.roi_extension = 0
    self.roi_scale_index_multiplier = 1.0
    
    half_image_size = max(item.args.OriginalSourceWidth, item.args.OriginalSourceHeight) // 2
    if item.args.RoIInflation == 'auto':
      percent = self.get_roi_extension_percent( item.args.quality )
      self.roi_extension = int(percent*half_image_size)
      self.roi_scale_index_multiplier = 1.0 - percent
    elif ("percent" in item.args.RoIInflation) or ("%" in item.args.RoIInflation):
      percent = float(item.args.RoIInflation.replace("percent","").replace("%","").strip()) / 100.0
      percent = max(0.0, min(1.0, percent))
      self.roi_extension = int(percent*half_image_size)
      self.roi_scale_index_multiplier = 1.0 - percent
    else:
      self.roi_extension = int(item.args.RoIInflation.strip())
      percent = self.roi_extension*1.0 / half_image_size
      percent = max(0.0, min(1.0, percent))
      self.roi_scale_index_multiplier = 1.0 - percent

    self.max_num_rois = item.args.RoIRetargetingMaxNumRoIs

    vcmrs.log(f"RoI parameters: accumulation_period:{self.accumulation_period} roi_update_period:{self.roi_update_period} desired_max_obj_size{self.desired_max_obj_size} network:{item.args.RoIGenerationNetwork}")
    vcmrs.log(f"max_num_rois:{self.max_num_rois} ArgRoIInflation:{item.args.RoIInflation} roi_extension:{self.roi_extension}")
    vcmrs.log(f"RoIAdaptiveMarginDilation:{item.args.RoIAdaptiveMarginDilation}")

    # moved this code below to here, from inside of generateRoI function
    if (item.args.RoIDescriptorMode=="load") and (item.args.RoIDescriptor is not None):
      self.mode_network_generate_descriptor = False
      self.mode_load_descriptor_from_file   = True
      self.mode_save_descriptor_to_file     = False
      self.mode_exit_after_roi              = False
    elif (item.args.RoIGenerationNetwork is not None):     
      self.mode_network_generate_descriptor = True
      self.mode_load_descriptor_from_file   = False
      self.mode_save_descriptor_to_file     = (item.args.RoIDescriptorMode=="save") or (item.args.RoIDescriptorMode=="saveexit")
      self.mode_exit_after_roi              = (item.args.RoIDescriptorMode=="saveexit")
    else:
      vcmrs.log(f"RoI processing: no source of descriptors! Specify RoIGenerationNetwork or RoIDescriptor parameter!")
      sys.exit(0)

    self.retargeting_enabled = "off" not in item.args.RoIRetargetingMode
    #self.retargeting_enabled = False
      
    self.retargeting_content_resolution_first = "first" in item.args.RoIRetargetingMode
    self.retargeting_content_resolution_sequence = "sequence" in item.args.RoIRetargetingMode
    self.retargeting_content_resolution_dynamic = "dynamic" in item.args.RoIRetargetingMode
    self.retargeting_content_resolution_pad = "pad" in item.args.RoIRetargetingMode
    self.retargeting_content_resolution_fit = "fit" in item.args.RoIRetargetingMode
    self.retargeting_content_resolution_user = None
    if "user" in item.args.RoIRetargetingMode:
      self.retargeting_content_resolution_user = item.args.RoIRetargetingMode[1:3]
      
    vcmrs.log(f"retargeting_content_resolution_first      {self.retargeting_content_resolution_first    }")
    vcmrs.log(f"retargeting_content_resolution_sequence   {self.retargeting_content_resolution_sequence }")
    vcmrs.log(f"retargeting_content_resolution_dynamic    {self.retargeting_content_resolution_dynamic  }")
    vcmrs.log(f"retargeting_content_resolution_pad        {self.retargeting_content_resolution_pad      }")
    vcmrs.log(f"retargeting_content_resolution_fit        {self.retargeting_content_resolution_fit      }")
    vcmrs.log(f"retargeting_content_resolution_user       {self.retargeting_content_resolution_user     }")
    
    if item._is_yuv_video:
      height = item.args.SourceHeight
      width = item.args.SourceWidth

      temp_folder_path_yuv = io_utils.create_temp_folder_suffix(os.path.abspath(os.path.join(item.args.working_dir,"temp_yuv")))
      temp_folder_path_out = io_utils.create_temp_folder_suffix(os.path.abspath(os.path.join(item.args.working_dir,"temp_out")))

      temp_file_name_template_yuv  = os.path.join(temp_folder_path_yuv, "frame_%06d.png")
      temp_file_name_template_out  = os.path.join(temp_folder_path_out, "frame_%06d.png")
    
      if self.mode_network_generate_descriptor or not USE_NATIVE_COLORSPACE:
        temp_folder_path_rgb = io_utils.create_temp_folder_suffix(os.path.abspath(os.path.join(item.args.working_dir,"temp_rgb")))
        rgb_file_name_template  = os.path.join(temp_folder_path_rgb, "frame_%06d.png")
      
        ## ffmpeg conversion - just for identical generation of descriptors
        pixfmt = "yuv420p" if item.args.InputBitDepth==8 else "yuv420p10le"
        ffmpeg_command = [item.args.ffmpeg, "-y", "-nostats", "-hide_banner", "-loglevel", "error" ]
        ffmpeg_command += ['-threads', '1'] 
        if item.args.InputBitDepth==8:
          ffmpeg_command += ["-f", "rawvideo"]
        ffmpeg_command += [
          "-s", f"{width}x{height}",
          "-pix_fmt", pixfmt,
          "-i", input_fname,
          "-vsync", "1",
          "-y", # duplicate "-y" ????
          "-start_number", "000000",
          "-pix_fmt", "rgb24",
          rgb_file_name_template]
        err = utils.start_process_expect_returncode_0(ffmpeg_command, wait=True)
        assert err==0, "Generating sequence in png format failed."
      
      else:
        temp_folder_path_rgb = None

      if USE_NATIVE_COLORSPACE:
        data_utils.convert_yuv_to_yuvpng(input_fname, item.args.SourceWidth, item.args.SourceHeight, item.args.InputChromaFormat, item.args.InputBitDepth,  temp_file_name_template_yuv, item.args.InputBitDepth)

      if USE_NATIVE_COLORSPACE:
        self.generateRoI(temp_folder_path_rgb, temp_folder_path_yuv, item.args.InputBitDepth, temp_file_name_template_out, item)
      else:
        self.generateRoI(temp_folder_path_rgb, None, 8, temp_file_name_template_out, item)

      if USE_NATIVE_COLORSPACE:
        data_utils.convert_yuvpng_to_yuv(temp_folder_path_out, item.args.InputBitDepth, output_fname, "420", item.args.InputBitDepth)
      else:

        pixfmt = "yuv420p" if item.args.InputBitDepth==8 else "yuv420p10le"
        ffmpeg_command = [
          item.args.ffmpeg, "-y", "-nostats", "-hide_banner", "-loglevel", "error",
          "-threads", "1", # PUT
          "-f", "image2",
          "-framerate", item.args.FrameRate,
          "-i", temp_file_name_template_out,
          "-pix_fmt", pixfmt,
          output_fname]
        err = utils.start_process_expect_returncode_0(ffmpeg_command, wait=True)
        assert err==0, "Generating sequence to yuv format failed."

      #while True:
      #  pass

      if temp_folder_path_rgb:
        shutil.rmtree(temp_folder_path_rgb)
      shutil.rmtree(temp_folder_path_yuv)
      shutil.rmtree(temp_folder_path_out)
      
    elif item._is_dir_video:
      makedirs(output_fname)
      vcmrs.log(f"Image InputBitDepth: {item.args.InputBitDepth}")
      self.generateRoI(input_fname, None, 8, os.path.join(output_fname, "frame_%06d.png"), item)
    elif self._is_image(input_fname):
      if os.path.exists(output_fname):
        if os.path.isdir(output_fname):
          shutil.rmtree(output_fname)
        else:
          os.remove(output_fname)
      io_utils.enforce_symlink(input_fname, output_fname)
      
  def update_spatial_parameter(self, item, spatial_resample_width, spatial_resample_height):    
    param_data = item.get_parameter('SpatialResample')
    if param_data is not None and len(param_data) != 1:
      param_data[1] = (spatial_resample_width >> 8) & 0xFF 
      param_data[2] = spatial_resample_width & 0xFF
      param_data[3] = (spatial_resample_height >> 8) & 0xFF 
      param_data[4] = spatial_resample_height & 0xFF  

################### moved outside so that other tools can use this:

def get_roi_accumulation_periods(args, IntraPeriod, FrameRateRelativeVsInput):

  if args.RoIAccumulationPeriod == 0: # default
    if args.Configuration == 'AllIntra':
      accumulation_period = 1
    elif args.Configuration == 'LowDelay':
      accumulation_period = 32 # defaults to CTC configuration which is 32 frames, indepedently from temporal resampling, also as originally in ROI plugin
    else:                
      accumulation_period = IntraPeriod # intra period, but adjusted to operation of temporal resampling (if active)
      
  elif args.RoIAccumulationPeriod<0: # negative: take value as absolute, not adjusted
    accumulation_period = abs(args.RoIAccumulationPeriod) 
  else: # postive: value, but adjusted to operation of temporal resampling (if active)
    accumulation_period = int( abs(args.RoIAccumulationPeriod) * FrameRateRelativeVsInput )
    
  accumulation_period = max(accumulation_period, 1) # must be >= 1
  
  roi_update_period = min( IntraPeriod, accumulation_period )
  roi_update_period = max(roi_update_period, 1) # must be >= 1

  return accumulation_period, roi_update_period
