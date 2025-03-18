# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import vcmrs
from numba import jit
from collections import deque
import torch
import sys
import re
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.kalman_filter import KalmanFilter
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.log import logger
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.models import *
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.tracker import matching
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.tracker.basetrack import BaseTrack, TrackState
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils import datasets as datasets
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.parse_config import parse_model_cfg
from vcmrs.ROI.roi_retargeting import roi_retargeting
from vcmrs.ROI.roi_syntax import roi_consts
from vcmrs.ROI.roi_utils import roi_utils
import os
import zlib
import base64
import json
import subprocess
import shutil
import cv2
import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from vcmrs.Utils import utils
from vcmrs.Utils.io_utils import makedirs
import vcmrs.ROI.roi_generator.context_aware_RoI_scailing as cars


class RoIImageGenerator(object):
  def __init__(self, opt, item, seqLength=0):
    self.opt = opt
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vcmrs.log(f"RoI generation with {self.device}")
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    if item.args.RoIGenerationNetwork == 'yolov3_1088x608':
      cfg_dict = parse_model_cfg(opt.cfg)
      self.img_size = [int(cfg_dict[0]["width"]), int(cfg_dict[0]["height"])]
      #dataloader = datasets.LoadImages(input_dir, self.img_size)
      #self.generate_RoI_YOLOv4(input_dir, item, dataloader, output_dir, self.accumulation_period)

      self.model = Darknet(opt.cfg, nID=14455)
      self.model.load_state_dict(torch.load(opt.weights, map_location=self.device)['model'], strict=False)
      self.model.to(self.device).eval()
    elif item.args.RoIGenerationNetwork == 'faster_rcnn_X_101_32x8d_FPN_3x':
      # setup detectron2 logger
      setup_logger()

      # create config
      cfg = get_cfg()
      
      cfg.merge_from_file(os.path.join(cur_dir_path, "./config/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      cfg.MODEL.WEIGHTS = os.path.join(cur_dir_path, "./weights/model_final_68b088.pkl")
      if self.device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu'
      # create predictor
      self.predictor = DefaultPredictor(cfg)
    
    self.item=item
    self.frame_id = 0
    self.objects_for_frames = []
    self.img_temp = []
    self.seqLength = seqLength
    self.descriptor = []
    self.frame_skip = 0 #item.args.FrameSkip  # instead of preserving FrameSkip - remove those frames, and set FrameSkip=0 (required for potential change of order between TR and ROI)
    
    
  def makeBinaryMask(self, img, input_dict, width, height):    
    objects = []
        
    for i in range(len(input_dict)):
      box = input_dict[i].cpu().numpy()
      x1 = int(box[0])
      x2 = int(box[2])
      y1 = int(box[1])
      y2 = int(box[3])
      objects.append([x1,y1,x2,y2])
      
    objects.sort(key = lambda x:(x[0], x[1]))
    widthThreshold = width/60
    heightThreshold = height/25
    
    for i in range(len(objects)-1):
      for j in range(i,len(objects)-i-1):
        if(abs(objects[j][2]-objects[j+1][0]) < widthThreshold or objects[j][2] > objects[j+1][0]):
          if(abs(objects[j][1]-objects[j+1][1]) < heightThreshold and abs(objects[j][3] - objects[j+1][3]) < heightThreshold):
            objects[j+1][0] = objects[j][0]
            objects[j+1][1] = objects[j][1] if objects[j][1] < objects[j+1][1] else objects[j+1][1]
            objects[j+1][3] = objects[j][3] if objects[j][3] > objects[j+1][3] else objects[j+1][3]
     
    return objects

  def roiAccumulation(self, objs_for_frames, img, bit_depth, width, height, start_idx, end_idx, feather=20): # equivalent of original proposal, code refactored

    binary_FG = np.full((height, width,3), (1<<(bit_depth-1)), dtype=img.dtype)  
    
    for j in range(start_idx, end_idx+1):
      for i in range(len(objs_for_frames[j])):
        bbox = objs_for_frames[j][i]
        x1, y1, x2, y2 = roi_utils.rect_extend_limited(bbox, feather, width, height)
      
        binary_FG[y1:y2, x1:x2,:] = img[y1:y2, x1:x2,:]
        
    return binary_FG
  
  def getMaskWithFeather(self, width, height, feather, value, start_idx, end_idx):
    return roi_utils.get_mask_with_feather(self.objectsext_for_frames, width, height, feather, value, start_idx, end_idx)

  def roiAccumulationRetargeting(self, img, bit_depth, retargeting_params, width, height, start_idx, end_idx):

    org_size_y, org_size_x, num_comps = img.shape
    
    org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = retargeting_params

    #deep_background_mode = "before_rtg"
    deep_background_mode = "after_rtg_64"
    
    if self.item.args.RoIRetargetingBackgroundScale<roi_consts.MAXIMAL_SCALE_FACTOR: # 15
      deep_background_mode = "" # background is preserved

    feather = 10
    slope_to_blur_size_half = 50 # feather*3        slope_to_blu_size_total = 120
    closing = 160
    deep_background_color = (1<<(bit_depth-1))

    mask_i = self.getMaskWithFeather(width, height, 5, 1, start_idx, end_idx)
    
    if self.item.args.RoIBackgroundFilter>=1.0:
      blur = img.copy() # no blur at all
    else:
      if self.item.args.RoIBackgroundFilter>0.001:
        #num_coeffs = 101
        num_coeffs = int(1.0/self.item.args.RoIBackgroundFilter)*2+1
        num_coeffs = min(301, max(11, num_coeffs))
        scale = 5
        blur = roi_utils.blur_background_scaled_no_aliasing(num_coeffs, mask_i, img, bit_depth, self.item.args.RoIBackgroundFilter*0.5, scale)
      else:
        scale = 5
        blur = roi_utils.blur_background_scaled(101, mask_i, img, bit_depth, 0.001*0.5*scale, scale)

    if deep_background_color=="average":    
      #deep_background_color = blur.mean(axis=0).mean(axis=0) # not numerically stable
      dy, dx, comps = blur.shape
      num_pixels = dx*dy
      deep_background_color = np.zeros( (comps), dtype=img.dtype)
      for c in range(comps):
        deep_background_color[c] = np.clip( (blur[:,:,c].sum(dtype=np.int64) + num_pixels//2) // num_pixels, 0, (1<<bit_depth)-1 )
    
    mask_slope = self.getMaskWithFeather(width, height, feather, 128, start_idx, end_idx)
    mix_slope = roi_utils.mix_images(blur, mask_slope, slope_to_blur_size_half, img)
    mask_closing = self.getMaskWithFeather(width, height, feather+closing, 128, start_idx, end_idx)
    mask_closing = roi_utils.erode_image(mask_closing, closing*2+1)
    
    if deep_background_mode == "before_rtg":
      deep_background = np.zeros((height, width,3),dtype=img.dtype)  
      deep_background[:,:,:] = deep_background_color
    
      img = roi_utils.mix_images(deep_background, mask_closing, 0, mix_slope)
    else: 
      img = mix_slope

    img = roi_retargeting.retarget_image(img, bit_depth, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y)
    
    if deep_background_mode.startswith("after_rtg") and (rtg_size_x!=0) and (rtg_size_y!=0): # bug-fix
      
      mask_closing3 = np.zeros( (org_size_y, org_size_x, num_comps), dtype=np.uint8) # 8bit -> it is just a mask
      mask_closing3[:,:,0] = mask_closing
      
      mask_closing3_rtg = roi_retargeting.retarget_image(mask_closing3, 8, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y)
            
      mask_closing_rtg = mask_closing3_rtg[:,:,0]
      
      spl = deep_background_mode.split("_")
      raster = int(spl[2])
      
      dx = rtg_size_x
      dy = rtg_size_y
      
      dxn = (dx+raster-1)//raster
      dyn = (dy+raster-1)//raster
      
      dxr = dxn * raster
      dyr = dyn * raster
      
      if (dx!=dxr) or (dy!=dyr):
        tmp = np.zeros( (dyr, dxr), dtype = np.uint8)
        tmp[0:dy, 0:dx] = mask_closing_rtg
 
        for x in range(dx, dxr):
          tmp[0:dy, x] = mask_closing_rtg[0:dy, dx-1]
            
        for y in range(dy, dyr):
          tmp[y, 0:dx] = mask_closing_rtg[dy-1, 0:dx]
                
        tmp[dy:, dx:] = mask_closing_rtg[dy-1, dx-1]
        
        mask_closing_rtg = tmp
            
      mask_closing_rtgs = cv2.resize(mask_closing_rtg, (dxn, dyn), interpolation=cv2.INTER_AREA)
      
      #mask_closing_rtgst = (mask_closing_rtgs > 64).astype(np.uint8)*128
      mask_closing_rtgst = (mask_closing_rtgs >= 2).astype(np.uint8)*128   # 1.5%
      
      mask_closing_rtgstr = cv2.resize(mask_closing_rtgst, (dxr, dyr), interpolation=cv2.INTER_AREA)
      mask_closing_rtgstr = mask_closing_rtgstr[0: rtg_size_y, 0:rtg_size_x]
      
      deep_background = np.zeros((rtg_size_y, rtg_size_x,3),dtype=img.dtype)  
      deep_background[:,:,:] = deep_background_color
      
      img = roi_utils.mix_images(deep_background, mask_closing_rtgstr, 0, img)

    if self.opt.retargeting_content_resolution_pad: # new +pad mode

      img2 = np.zeros( (org_size_y,org_size_x,num_comps), dtype=img.dtype )
      img2[:,:,:] = 1<<(bit_depth-1)
      img2[0:rtg_size_y, 0:rtg_size_x, :] = img[0:rtg_size_y, 0:rtg_size_x :]
            
      #fill_x = rtg_size_x > org_size_x * 0.75
      #fill_y = rtg_size_y > org_size_y * 0.75

      inner_codec_CTU_alignment_size = 64

      fill_x = (org_size_x - rtg_size_x) < inner_codec_CTU_alignment_size
      fill_y = (org_size_y - rtg_size_y) < inner_codec_CTU_alignment_size
            
      if fill_x:
        for x in range(rtg_size_x, org_size_x):
          img2[0:rtg_size_y, x, :] = img[0:rtg_size_y, rtg_size_x-1, :]
            
      if fill_y:
        for y in range(rtg_size_y, org_size_y):
          img2[y, 0:rtg_size_x, :] = img[rtg_size_y-1, 0:rtg_size_x, :]
                
      if fill_x and fill_y:
        img2[rtg_size_y:, rtg_size_x:, :] = img[rtg_size_y-1, rtg_size_x-1, :]
            
      img = img2

    return img

  def networkDetectRoI(self, img_rgb):
    
    if self.item.args.RoIGenerationNetwork == 'yolov3_1088x608':  
      
      # Padded resize
      im_blob, _, _, _ = datasets.letterbox(img_rgb, height=self.img_size[1], width=self.img_size[0])

      # Normalize RGB
      im_blob = im_blob[:, :, ::-1].transpose(2, 0, 1)
      im_blob = np.ascontiguousarray(im_blob, dtype=np.float32)
      im_blob /= 255.0
      
      im_blob = torch.from_numpy(im_blob).to(self.opt.device).unsqueeze(0)
    
      with torch.no_grad():
        pred = self.model(im_blob)

      pred = pred[pred[:, :, 4] > self.opt.conf_thres]

      if len(pred) > 0:
        dets1 = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].to(self.device)

        scale_coords(self.img_size, dets1[:, :4], img_rgb.shape).round()

        det2 = [tlbrs[:4] for (tlbrs, f) in zip(dets1[:, :5], dets1[:, 6:])]
      
        return self.makeBinaryMask(img_rgb, det2, img_rgb.shape[1], img_rgb.shape[0])
    
    elif self.item.args.RoIGenerationNetwork == 'faster_rcnn_X_101_32x8d_FPN_3x':
      # read image

      # make prediction
      outputs = self.predictor(img_rgb)
      
      # get predictions
      instances = outputs["instances"]
      det2 = instances.pred_boxes.tensor
    
      return self.makeBinaryMask(img_rgb, det2, img_rgb.shape[1], img_rgb.shape[0])
    
    return [[0,0,0,0,]]

  def calculate_iou(self, gt, pred):
    """Computing IoU between GT and Pred."""
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])
    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])
    
    # Calculating the intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of each box
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    
    # Calculating the union area
    union = gt_area + pred_area - intersection
    
    # Compute and return IoU
    return intersection / union if union > 0 else 0


  def calculate_average_iou(self, gt_boxes, pred_boxes):
    """Compute the average IoU between GT boxes and Pred boxes."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
      return 0  # If GT or Pred is empty, the average IoU is 0.
    
    total_iou = 0
    count = 0
    
    for gt in gt_boxes:
      for pred in pred_boxes:
        total_iou += self.calculate_iou(gt, pred)
        count += 1
    
    return total_iou / count if count > 0 else 0

  def appendMariginsInfoForObjectsForFrame(self, objects, img_rgb): # processes objects in place
    shape = img_rgb.shape
    
    scaling_method = 1
    scaled_objects = cars.scale_rois([objects], scaling_method, image_size=shape)  # scale_rois operates  on list of frames, so work with list of 1 frame
    scaled_objects = scaled_objects[0]
      
    no_margin_objects = self.networkDetectRoI(self.roiAccumulation([objects], img_rgb, 8, shape[1], shape[0], 0,0))
    margin_objects = self.networkDetectRoI(self.roiAccumulation([scaled_objects], img_rgb, 8, shape[1], shape[0], 0,0))
      
    margin_iou = self.calculate_average_iou(objects, margin_objects)
    no_margin_iou = self.calculate_average_iou(objects, no_margin_objects)

    if margin_iou > no_margin_iou:
      objects.append(f'"scaling_method={scaling_method}"')
      vcmrs.log(f"RoI processing: append: scaling_method={scaling_method}")
    else:
      objects.append(f'"scaling_method=0"')
      vcmrs.log("RoI processing: append: original")

  def extractMariginsInfoForObjectsForFrame(self, objects): # processes objects in place
    scaling_method = None
    for i in range(len(objects)-1,-1,-1):
      obj = objects[i]
      if isinstance(obj, str):
        if obj.startswith("scaling_method="):
          scaling_method = int(obj[15:])
        del objects[i]
    return scaling_method

  def applyMariginsForObjectsForFrames(self, objects_for_frames, shape): # processes objects_for_frames in place
  
    for i in range(len(objects_for_frames)):
      objects = objects_for_frames[i]
      scaling_method = self.extractMariginsInfoForObjectsForFrame(objects)
      #vcmrs.log(f"RoI processing: apply: scaling_method={scaling_method}")
      if scaling_method>0:
        scaled_objects = cars.scale_rois([objects], scaling_method, image_size=shape)  # scale_rois operates  on list of frames, so work with list of 1 frame
        objects_for_frames[i] = scaled_objects[0]
          
  def generateRoIImage(self, roi_update_period, accumulation_period, desired_max_obj_size, max_num_rois, img_rgb, img_process, bit_depth, save_name_template):
    if self.frame_id % 20 == 0:
      vcmrs.log(f"RoI processing: frame {self.frame_id} accumulation_period{accumulation_period} roi_update_period{roi_update_period}")

    # moved code from here to the outside, to have broader access through "self.opt." scope

    if self.opt.mode_network_generate_descriptor:
      # use networks to generate
      objects = self.networkDetectRoI(img_rgb)

      #if self.item.args.RoIAdaptiveMarginDilation:
      if True: # always append information, decide whether to use -> later
        self.appendMariginsInfoForObjectsForFrame(objects, img_rgb)

      self.objects_for_frames.append(objects)

    #img_process = img_rgb
    self.img_temp.append(img_process)
    
    # wait until end of sequence and process img_temp
    #if self.frame_id == 1: # second frame, for testing
    if self.frame_id == self.seqLength-1:
    
      if self.opt.mode_load_descriptor_from_file:
        vcmrs.log(f"RoI processing: loading descriptor {self.item.args.RoIDescriptor}")
        objects_for_frames_dict = roi_utils.load_descriptor_from_file( self.item.args.RoIDescriptor )
        
        self.objects_for_frames = []
        for org_idx in self.item.OriginalFrameIndices[self.item.args.FrameSkip:] :
          try:
            self.objects_for_frames.append( objects_for_frames_dict[org_idx] )
          except IndexError as e:
            vcmrs.log(f"RoI processing: ERROR while fetching descriptor for frame: {org_idx}")
            vcmrs.log(f"RoI processing: OriginalFrameIndices: {self.item.OriginalFrameIndices}")
            vcmrs.log(f"RoI processing: FrameSkip: {self.item.args.FrameSkip}")
            raise

      if self.opt.mode_save_descriptor_to_file:
        descriptor_file = self.item.args.RoIDescriptor
        if descriptor_file is None:
          descriptor_path = os.path.join(self.item.working_dir, 'RoIDescriptor')
          makedirs(descriptor_path)
          descriptor_file = os.path.basename(self.item.fname) + ".txt"
          descriptor_file = re.sub(r'_qp\d+', '', descriptor_file)
          descriptor_file = os.path.join(descriptor_path,descriptor_file)
            
        if os.path.isfile(descriptor_file):
          for i in range(1,1000):
            test = descriptor_file+"("+str(i)+")"
            if not os.path.isfile(test):
              break
          descriptor_file = test
            
        vcmrs.log(f"RoI processing: saving descriptor {descriptor_file}")
        
        # save... (and exit later maybe?)
        
        objects_for_frames_dict = {}
        for local_idx, org_idx in enumerate(self.item.OriginalFrameIndices[self.item.args.FrameSkip:]):
          objects_for_frames_dict[org_idx] = self.objects_for_frames[local_idx]
          
        roi_utils.save_descriptor_to_file(descriptor_file, objects_for_frames_dict)

      if self.opt.mode_exit_after_roi:
        vcmrs.log(f"RoI processing: exitting after saving")
        vcmrs.log(f"Encoding completed in 0 seconds")
        sys.exit(0)
    
      vcmrs.log(f"RoI processing: saving images")

      width, height = img_process.shape[1], img_process.shape[0] # last frame in the sequence - could be any just to grab resolution of the input video

      vcmrs.log(f"RoI processing: roi_extension: {self.opt.roi_extension}")    
      vcmrs.log(f"RoI processing: roi_scale_index_multiplier: {self.opt.roi_scale_index_multiplier}")    
      
      # apply scaling or not, depending on "scaling_method" flag for each frame
      if self.item.args.RoIAdaptiveMarginDilation:
        self.applyMariginsForObjectsForFrames(self.objects_for_frames, self.img_temp[0].shape)
      else:
        # remove "scaling_method" flag - not needed
        for objects in self.objects_for_frames:
          self.extractMariginsInfoForObjectsForFrame(objects)

      self.objectsext_for_frames = roi_utils.extend_objects(self.objects_for_frames, self.opt.roi_extension, width, height)

      if self.opt.retargeting_enabled:
        
        num_roi_update_periods =  (self.seqLength + roi_update_period -1 ) // roi_update_period # round up
        ip_rois = []
        ip_resolutions = [] # possible different resolutions per each GOP
        
        #feather = 20
        feather = 5

        for ip in range(num_roi_update_periods):
          start_idx = ip * roi_update_period
          end_idx = min(start_idx + roi_update_period, self.seqLength)-1

          start_idx_win = max(0, start_idx - self.item.args.RoIAccumulationWindowExtension) # gather ROI with extended window
          rois = roi_retargeting.find_ROIs(self.objects_for_frames, feather, desired_max_obj_size, width, height, start_idx_win, end_idx, None)

          if len(rois)>max_num_rois:
            rois = roi_retargeting.find_ROIs_simple(self.objects_for_frames, feather, width, height, start_idx_win, end_idx)

          rois = roi_utils.extend_rois_and_improve_scale(rois, self.opt.roi_extension, width, height, self.opt.roi_scale_index_multiplier)

          rois.sort(key=lambda tup: tup[0], reverse = True)  # maintain order of RoIs according to scale
          if self.item.args.RoIRetargetingBackgroundScale<roi_consts.MAXIMAL_SCALE_FACTOR: # 15 
            rois = roi_retargeting.include_background_ROI(rois, width, height, self.item.args.RoIRetargetingBackgroundScale) # background is preserved as first RoI
          
          ip_rois.append( rois )
        
        align_size = 64
        rtg_size_x_final = 0
        rtg_size_y_final = 0
        for ip in range(num_roi_update_periods):
          org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y =  roi_retargeting.generate_coords_even(width, height, ip_rois[ip], align_size, align_size, None, None)
          #org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y =  roi_retargeting.generate_coords(width, height, ip_rois[ip], align_size, align_size, None, None)
          #vcmrs.log(f"RoI processing: ip: {ip:02d} would suggest resolution:    {width}x{height} -> {rtg_size_x}x{rtg_size_y}")
          
          if ((ip==0) and (self.opt.retargeting_content_resolution_first)) or (self.opt.retargeting_content_resolution_sequence): # new resolution control modes
            rtg_size_x_final = max(rtg_size_x_final, rtg_size_x)
            rtg_size_y_final = max(rtg_size_y_final, rtg_size_y)
                   
        vcmrs.log(f"RoI processing: retargeting: Resolution suggested:  {width}x{height} -> {rtg_size_x_final}x{rtg_size_y_final}")

        if self.opt.retargeting_content_resolution_user is not None: # new resolution control modes
          rtg_size_x_final = self.opt.retargeting_content_resolution_user[0]
          rtg_size_y_final = self.opt.retargeting_content_resolution_user[1]
        
        rtg_size_x_final = max(rtg_size_x_final, width//16) # do not shrink more than 1:16
        rtg_size_y_final = max(rtg_size_y_final, height//16)
        
        rtg_size_x_final = max(rtg_size_x_final, 1) # do not shrink more than to 1x1 pixels
        rtg_size_y_final = max(rtg_size_y_final, 1)
        
        rtg_size_x_final = (rtg_size_x_final+1) & ~1 # even number of pixels (4:2:0)
        rtg_size_y_final = (rtg_size_y_final+1) & ~1 # even number of pixels (4:2:0)
        
        rtg_size_x_final = min(rtg_size_x_final, width) # do not extend (possible due to alignment, etc.)
        rtg_size_y_final = min(rtg_size_y_final, height)
        
        rtg_size_x_final = max(rtg_size_x_final, 128) # for LIC intra codec
        rtg_size_y_final = max(rtg_size_y_final, 128)
        # new resolution control modes
        if self.opt.retargeting_content_resolution_dynamic and self.opt.retargeting_content_resolution_fit:
          rtg_size_x_final = width
          rtg_size_y_final = height

        update_size_x = rtg_size_x_final
        update_size_y = rtg_size_y_final

        if self.opt.retargeting_content_resolution_pad or self.opt.retargeting_content_resolution_fit:
          update_size_x = None
          update_size_y = None

        if self.opt.retargeting_content_resolution_dynamic and self.opt.retargeting_content_resolution_pad:
          rtg_size_x_final = None
          rtg_size_y_final = None
        
        vcmrs.log(f"RoI processing: retargeting: Resolution final:      {width}x{height} -> {rtg_size_x_final}x{rtg_size_y_final}")

        retargeting_params_for_ip = []
        for ip in range(num_roi_update_periods):
          res = roi_retargeting.generate_coords_even(width, height, ip_rois[ip], None, None, rtg_size_x_final, rtg_size_y_final)
          #res = roi_retargeting.generate_coords(width, height, ip_rois[ip], None, None, rtg_size_x_final, rtg_size_y_final)
          org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = res
          rtg_size_x = min(rtg_size_x, width) # do not extend (possible due to alignment, etc.)
          rtg_size_y = min(rtg_size_y, height)
          rtg_size = (rtg_size_x, rtg_size_y)
          ip_resolutions.append( rtg_size ) # possible different resolutions per each GOP
          retargeting_params_for_ip.append(res)
        
        # just print for debug
        for ip in range(num_roi_update_periods):
          res = retargeting_params_for_ip[ip]
          org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = res
      
        num_accumulation_periods = (self.seqLength + accumulation_period -1 ) // accumulation_period # round up
        for ap in range(num_accumulation_periods):
          start_idx = ap * accumulation_period
          end_idx = min(start_idx + accumulation_period, self.seqLength)-1
          start_idx_win = max(0, start_idx - self.item.args.RoIAccumulationWindowExtension) # accumulate gray with extended window
          for f in range(start_idx, end_idx+1):
            img = self.roiAccumulationRetargeting(self.img_temp[f], bit_depth, retargeting_params_for_ip[f // roi_update_period], width, height, start_idx_win, end_idx)
            cv2.imwrite(save_name_template % (f+self.frame_skip),img) 

      else: # no retargetting - Original Code (refactored)

        if self.seqLength <= accumulation_period:
          for i in range(self.seqLength):
            img_save = self.roiAccumulation(self.objectsext_for_frames, self.img_temp[i], bit_depth, width, height, 0, self.seqLength-1)
            cv2.imwrite( save_name_template % (i+self.frame_skip) ,img_save) 
        else:
          for i in range(self.seqLength):
            if (i+1) % accumulation_period == 0:
              if (i+accumulation_period) > (self.seqLength-1):
                for j in range(i-(accumulation_period-1),i+1):
                  start_idx = i-(accumulation_period-1)
                  start_idx_win = max(0, start_idx - self.item.args.RoIAccumulationWindowExtension) # accumulate gray with extended window
                  img_save = self.roiAccumulation(self.objectsext_for_frames, self.img_temp[j], bit_depth, width, height, start_idx_win, i)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save) 
                for j in range(i+1,self.seqLength):
                  start_idx = i
                  start_idx_win = max(0, start_idx - self.item.args.RoIAccumulationWindowExtension) # accumulate gray with extended window
                  img_save = self.roiAccumulation(self.objectsext_for_frames, self.img_temp[j], bit_depth, width, height, start_idx_win, self.seqLength-1)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save)
              else:
                for j in range(i-(accumulation_period-1),i+1):
                  start_idx = i-(accumulation_period-1)
                  start_idx_win = max(0, start_idx - self.item.args.RoIAccumulationWindowExtension) # accumulate gray with extended window
                  img_save = self.roiAccumulation(self.objectsext_for_frames, self.img_temp[j], bit_depth, width, height, start_idx_win, i)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save) 

      roi_size_limit=self.item.args.nnlf_roi_size_limit_RA # RA
      if self.item.args.Configuration=="LowDelay":
        roi_size_limit=self.item.args.nnlf_roi_size_limit_LD
      elif self.item.args.Configuration=="AllIntra":
        roi_size_limit=self.item.args.nnlf_roi_size_limit_AI
        
      if self.item.args.NnlfSwitch == 'NnlfSliceBased':
        self.item.args.NnlfSwitchEnable = 'Enabled'
        roi_utils.determine_roi_fallback(self.objectsext_for_frames, self.item.args.working_dir, width, height, roi_size_limit, limit_min=self.item.args.nnlf_roi_fallback_min, limit_max=self.item.args.nnlf_roi_fallback_max, trd_enc_fname=self.item.trd_enc_fname, TemporalResamplingPostHintFlag=self.item.args.TemporalResamplingPostHintFlag, PostHintSelectThreshold=self.item.args.PostHintSelectThreshold)

      vcmrs.log(f"RoI processing: done")
      
      if self.opt.retargeting_enabled:
        # possible different resolutions per each GOP
        return update_size_x, update_size_y,     rtg_size_x_final, rtg_size_y_final,     ip_rois, ip_resolutions
      return None, None,     None, None,     None, None
      
    self.frame_id += 1
    return None, None,     None, None,     None, None
  
