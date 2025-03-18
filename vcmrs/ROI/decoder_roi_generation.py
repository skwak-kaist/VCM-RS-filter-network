# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import cv2
import vcmrs
import shutil
import numpy as np
from itertools import count
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.ROI.roi_syntax import roi_bitdec
from vcmrs.ROI.roi_retargeting import roi_retargeting

USE_NATIVE_COLORSPACE = True

# component base class
class roi_generation(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    #self.log = print
    self.log = vcmrs.log
    
  def reverseRetargeting(self, inp_dir, out_dir, bit_depth, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions): # possible different resolutions per each GOP
  
    if not os.path.isdir(inp_dir):
      assert False, "Input file should be directory!"
    inp_fnames = sorted(glob.glob(os.path.join(inp_dir, '*.png')))
    
    seq_length = len(inp_fnames)
    
    num_roi_update_periods =  (seq_length + roi_update_period -1 ) // roi_update_period # round up
    retargeting_params_for_ip = []
    for ip in range(num_roi_update_periods):
      rtg_res = retargeting_gops_rtg_resolutions[ip]  # possible different resolutions per each GOP
      res = roi_retargeting.generate_coords(org_image_size_x, org_image_size_y, retargeting_gops_rois[ip], None, None, rtg_res[0], rtg_res[1])
      org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = res
      retargeting_params_for_ip.append(res)
    
    for f in range(seq_length):
      inp_fname = inp_fnames[f]
      basename = os.path.basename(inp_fname)
      out_fname = os.path.join(out_dir, basename)      
      ip = f // roi_update_period
      org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = retargeting_params_for_ip[ip]
      
      if USE_NATIVE_COLORSPACE:
        img = cv2.imread(inp_fname, cv2.IMREAD_UNCHANGED) # YVU
      else:
        img = cv2.imread(inp_fname) # BGR

      if len(img.shape)>2: # remove 4-th component if exists in the file
        if img.shape[2]>3:
          img = img[:,:,0:3]
        
      assert img is not None, f"Cannot read file:{inp_fname}"
        
      #cv2.imwrite(f"dec{f}a.png", img)
      img = roi_retargeting.retarget_image(img, bit_depth, rtg_coords_x, rtg_coords_y, org_coords_x, org_coords_y)
      cv2.imwrite(out_fname, img)
      #cv2.imwrite(f"dec{f}b.png", img)
      
    return

  def get_chroma_coords(self, scale, coords):
    out_coords = []
    for coord in coords:
        val = int(coord) >> scale
        out_coords.append(val)
    return out_coords

  def get_chroma_coords_all(self, scaleX, scaleY, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y):
     chroma_org_coords_x = self.get_chroma_coords(scaleX, org_coords_x)
     chroma_org_coords_y = self.get_chroma_coords(scaleY, org_coords_y)
     chroma_rtg_coords_x = self.get_chroma_coords(scaleX, rtg_coords_x)
     chroma_rtg_coords_y = self.get_chroma_coords(scaleY, rtg_coords_y)
     #vcmrs.log(f'{len(chroma_org_coords_x)} {len(chroma_org_coords_y)} {len(chroma_rtg_coords_x)} {len(chroma_rtg_coords_y)}')
     return chroma_org_coords_x, chroma_org_coords_y, chroma_rtg_coords_x, chroma_rtg_coords_y

  def reverseRetargetingYUV420(self, input_fname, output_fname, bit_depth, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions, width, height, chromaFormat):
    bytes_per_pixel = 2 if bit_depth > 8 else 1
    dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
    scaleX = 0
    scaleY = 0
    # Determine sizes based on format
    if chromaFormat == "420":
      y_size = width * height
      uv_size = (width // 2) * (height // 2)
      uv_shape = (height // 2, width // 2)
      scaleX = 1
      scaleY = 1
    elif chromaFormat == "422": # This is suported by the VCM-RS software but not included in the VCM specification
      y_size = width * height
      uv_size = (width // 2) * height
      uv_shape = (height, width // 2)
      scaleX = 1
    elif chromaFormat == "444":  # This is suported by the VCM-RS software but not included in the VCM specification
      y_size = width * height
      uv_size = width * height
      uv_shape = (height, width)
    else:
      raise ValueError("Unsupported chroma format: {}".format(chromaFormat))

    retargeting_params_for_ip = []
    for ip in range(len(retargeting_gops_rois)):
      rtg_res = retargeting_gops_rtg_resolutions[ip]  # possible different resolutions per each GOP
      res = roi_retargeting.generate_coords_even(org_image_size_x, org_image_size_y, retargeting_gops_rois[ip], None, None, rtg_res[0], rtg_res[1])
      retargeting_params_for_ip.append(res)

    with open(input_fname, 'rb') as input_file:
      makedirs(os.path.dirname(output_fname))
      with open(output_fname, 'wb') as output_file:
        for frame_idx in count():
          # read Y, U, V
          y_buffer = input_file.read(y_size * bytes_per_pixel)
          if len(y_buffer) == 0:  # check if at the end of the file
            break

          y = np.frombuffer(y_buffer, dtype=dtype).reshape((height, width))
          u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)
          v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)

          ip_idx = frame_idx // roi_update_period
          org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = retargeting_params_for_ip[ip_idx]
          chroma_org_coords_x, chroma_org_coords_y, chroma_rtg_coords_x, chroma_rtg_coords_y = self.get_chroma_coords_all(scaleX, scaleY, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y)
          yout = roi_retargeting.retarget_image_onecomp(y, bit_depth, rtg_coords_x, rtg_coords_y, org_coords_x, org_coords_y)
          uout = roi_retargeting.retarget_image_onecomp(u, bit_depth, chroma_rtg_coords_x, chroma_rtg_coords_y, chroma_org_coords_x, chroma_org_coords_y)
          vout = roi_retargeting.retarget_image_onecomp(v, bit_depth, chroma_rtg_coords_x, chroma_rtg_coords_y, chroma_org_coords_x, chroma_org_coords_y)

          output_file.write(yout.tobytes())
          output_file.write(uout.tobytes())
          output_file.write(vout.tobytes())

    return

  def process(self, input_fname, output_fname, item, ctx):
    # the default implementation is a bypass component

    try:
      gops_roi_bytes = item.get_parameter('ROI')
    except:
      gops_roi_bytes = None

    if gops_roi_bytes is not None:
      roi_update_period, rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions = roi_bitdec.decode_roi_params(gops_roi_bytes, self.log) # possible different resolutions per each GOP      
          
      item.args.SourceWidth = org_image_size_x
      item.args.SourceHeight = org_image_size_y
      H,W,C = item.video_info.resolution
      item.video_info.resolution = (org_image_size_y, org_image_size_x, C)
      
      if item._is_dir_video: # input format is directory with separate files           
        
        makedirs(output_fname)
        self.reverseRetargeting(input_fname, output_fname, 8, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions) # possible different resolutions per each GOP
      elif item._is_yuv_video: # input format is yuv file,  convert to pngs and back
        if True:
          bit_depth = 10
          self.reverseRetargetingYUV420(input_fname, output_fname, bit_depth, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions, W, H, "420") # possible different resolutions per each GOP
        else: 
          pngpath_before = output_fname+".tmp_dir_in"
          pngpath_after = output_fname+".tmp_dir_out"
        
          makedirs(pngpath_before)
          makedirs(pngpath_after)
        
        
          tmp_file_name_template = os.path.join(pngpath_before, "frame_%06d.png")
        
          if USE_NATIVE_COLORSPACE:
            data_utils.convert_yuv_to_yuvpng(input_fname, W, H, "420", 10,  tmp_file_name_template, 10)
          else:
            cmd = [
              item.args.ffmpeg, '-y', '-nostats', '-hide_banner', '-loglevel', 'error',
              '-threads', '1', # PUT
              '-f', 'rawvideo',
              '-s', f'{W}x{H}',
              '-pix_fmt', 'yuv420p10le',
              '-i', input_fname,
              '-vsync', '0',
              '-y',
              '-pix_fmt', 'rgb24', 
              tmp_file_name_template ] 
          
            err = utils.start_process_expect_returncode_0(cmd, wait=True)
            assert err==0, "Generating sequence in YUV format failed."

          bit_depth = 10 if USE_NATIVE_COLORSPACE else 8
          self.reverseRetargeting(pngpath_before, pngpath_after, bit_depth, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois, retargeting_gops_rtg_resolutions) # possible different resolutions per each GOP


          if USE_NATIVE_COLORSPACE:
            data_utils.convert_yuvpng_to_yuv(pngpath_after, 10, output_fname, "420", 10)
          else:
            cmd = [
              item.args.ffmpeg, '-y', '-nostats', '-hide_banner', '-loglevel', 'error',
              '-threads', '1', # PUT
              '-i', os.path.join(pngpath_after, 'frame_%06d.png'),
              '-f', 'rawvideo',
              '-pix_fmt', 'yuv420p10le',
              output_fname] 
            err = utils.start_process_expect_returncode_0(cmd, wait=True)                    
            assert err==0, "Generating sequence in YUV format failed."
        
          shutil.rmtree(pngpath_before)
          shutil.rmtree(pngpath_after)
        
      elif os.path.isfile(input_fname): 
        os.remove(output_fname)      
        makedirs(os.path.dirname(output_fname))
        enforce_symlink(input_fname, output_fname)
      else:
        assert False, f"Input file {input_fname} is not found"
      
      return     
      
    # else: retargeting is off:
    
    #if item._is_dir_video:
    if os.path.isdir(input_fname): #item._is_dir_video:
      # video data in a directory
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
        if os.path.isfile(output_frame_fname): os.remove(output_frame_fname)
        makedirs(os.path.dirname(output_frame_fname))
        enforce_symlink(fname, output_frame_fname)
    else:
      # image data or video in yuv format
      if os.path.isfile(output_fname): os.remove(output_fname)
      makedirs(os.path.dirname(output_fname))
      enforce_symlink(input_fname, output_fname)


