# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
import vcmrs
from vcmrs.Utils import utils
import datetime
import numpy as np
import cv2
from . import sr_interpolation
import time

from . import adaptivedownsample_data
import re

class AdaptiveDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)
    self.id_scale_factor_mapping = adaptivedownsample_data.id_scale_factor_mapping

  def process(self, input_fname, output_fname, item, ctx, chromaFormat="420", bit_depth=10):
    self.temporal_post_hint_process(item, input_fname)
    vcmrs.log("==========================================================decoder spatial resampling process start!==================================================================================")
    
    spatial_resample_width, spatial_resample_height, spatial_resample_filter_idx = self._get_parameter(item)

    if spatial_resample_filter_idx==0:
      upsample_filter_info=1

    if not spatial_resample_width or not spatial_resample_height or item._is_dir_video or not item._is_yuv_video:
      if item._is_dir_video:
        # video dir data
        os.makedirs(output_fname, exist_ok=True)
        fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
        for idx, fname in enumerate(fnames):
          output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")

      elif os.path.isfile(input_fname): 
        # image or YUV data
        makedirs(os.path.dirname(output_fname))
      else:
        raise FileNotFoundError(f"Input {input_fname} is not found.")

      if os.path.isfile(output_fname): os.remove(output_fname)
      enforce_symlink(input_fname, output_fname)
      return
    
    # Remove deprecated code (bicubic placeholder)
    # if item.args.Spatial_scalefactor_dir :
    #   filename = get_filename(item, input_fname)
    #   scale_list_file = os.path.join(item.args.Spatial_scalefactor_dir, f'{filename}.txt')
    #   with open(scale_list_file, 'r') as f:
    #       lines = f.readlines() 
    #   sc_list = [int(line.strip()) for line in lines]
    #   vcmrs.log(f'Spatial Resample bicubic debug mode : scale_list_file : {scale_list_file}\n{sc_list}')
    
     # the default implementation is a bypass component
    if os.path.isfile(input_fname):
        bytes_per_pixel = 2 if bit_depth > 8 else 1
        dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
        scaleX = 0
        scaleY = 0
        # Determine sizes based on format
        if chromaFormat == "420":
          uv_size = (spatial_resample_width // 2) * (spatial_resample_height // 2)
          uv_shape = (spatial_resample_height // 2, spatial_resample_width // 2)
          scaleX = 1
          scaleY = 1
        elif chromaFormat == "422":
          uv_size = (spatial_resample_width // 2) * spatial_resample_height
          uv_shape = (spatial_resample_height, spatial_resample_width // 2)
          scaleX = 1
        elif chromaFormat == "444":
          uv_size = spatial_resample_width * spatial_resample_height
          uv_shape = (spatial_resample_height, spatial_resample_width)
        else:
          raise ValueError("Unsupported chroma format: {}".format(chromaFormat))

        out_image_luma = np.zeros( (spatial_resample_height, spatial_resample_width), dtype = np.int16 )
        out_image_chroma = np.zeros( uv_shape, dtype = np.int16 )
        if spatial_resample_filter_idx == 0:
          start_time = time.time() #component coding time
          with open(input_fname, 'rb') as input_file:
            makedirs(os.path.dirname(output_fname))
            with open(output_fname, 'wb') as output_file:
              for frame_idx in range(len(item.video_info.ic_dec_size_info)):
              # read Y, U, V
                wl = item.video_info.ic_dec_size_info[frame_idx][0]
                hl = item.video_info.ic_dec_size_info[frame_idx][1]
                wc = wl >> scaleX
                hc = hl >> scaleY
                y_buffer = input_file.read(spatial_resample_width * spatial_resample_height * bytes_per_pixel)
                if len(y_buffer) == 0:  # check if at the end of the file
                  break
                ytmp = np.frombuffer(y_buffer, dtype=dtype).reshape((spatial_resample_height, spatial_resample_width))
                utmp = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)
                vtmp = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)
                y = ytmp[:hl, :wl]
                u = utmp[:hc, :wc]
                v = vtmp[:hc, :wc]
 
                if wl == spatial_resample_width and hl == spatial_resample_height:
                  output_file.write(y.tobytes())
                  output_file.write(u.tobytes())
                  output_file.write(v.tobytes())
                else:
                  sr_interpolation.InterpolationNumbaInt12tapLuma(out_image_luma, y)
                  output_file.write(out_image_luma.tobytes())
                  sr_interpolation.InterpolationNumbaInt6tapChroma(out_image_chroma, u, scaleX, scaleY)
                  output_file.write(out_image_chroma.tobytes())
                  sr_interpolation.InterpolationNumbaInt6tapChroma(out_image_chroma, v, scaleX, scaleY)
                  output_file.write(out_image_chroma.tobytes())
            process_time_duration = time.time() - start_time
            vcmrs.log(f"12tap/6tap sr process was applied. Time: {process_time_duration}")
        else:
           raise ValueError(f"Unsupported filter index (spatial_resample_filter_idx = {spatial_resample_filter_idx})")
        return

    else:
        vcmrs.debug("Sample interpolation process was not performed")
        if os.path.isfile(output_fname): os.remove(output_fname)
        enforce_symlink(input_fname, output_fname)
        #raise FileNotFoundError(f"Input {input_fname} is not found.")
        
  def _get_parameter(self, item):
    # sequence level parameter
    spatial_resample_width = None
    spatial_resample_height = None
    spatial_resample_filter_idx = None
    # sc_len = None
    # sc_list = None
    param_data = item.get_parameter('SpatialResample')
    vcm_spatial_resampling_flag = param_data[0]
    if vcm_spatial_resampling_flag :
      spatial_resample_width = (param_data[1] << 8) + param_data[2]
      spatial_resample_height = (param_data[3] << 8) + param_data[4]
      spatial_resample_filter_idx = param_data[5]
    return spatial_resample_width, spatial_resample_height, spatial_resample_filter_idx

  def temporal_post_hint_process(self, item, input_fname):
    temporal_post_hint_flag = 0
    if hasattr(item, 'trd_enc_fname'):
      cfg_path = item.trd_enc_fname
      if os.path.exists(cfg_path):
        with open(cfg_path, 'r+') as f:
          lines = f.readlines()
        new_lines = [i.strip() for i in lines if i.strip() != ""]
        for i in range(len(new_lines)):
          if ":" in new_lines[i]:
            key, value = new_lines[i].split(':')
            key = key.strip()
            value = value.strip()
            if key in ["TemporalResamplingPostHintFlag"]:
              temporal_post_hint_flag = int(value.strip())                   
              if temporal_post_hint_flag: 
                from vcmrs.TemporalResample import resample_score
                import torch
                import time
                num_pivot = item.args.FramesToBeEncoded
                with torch.no_grad():
                  fr_height,fr_width, _ = item.video_info.resolution
                  temporalassesser = resample_score._TemporalAssesser_(fr_height, fr_width)    
                  score_path = os.path.dirname(os.path.abspath(item.working_dir))
                  _ = temporalassesser.process_yuvs_assessment(input_fname, '420', 10, q_bit_depth=7, frame_num=num_pivot, call_position='dec', path=score_path)

def get_frame_from_spatial_resampled_video(fname, \
    W, 
    H,
    start_pos, 
    chroma_format='420', 
    bit_depth=8, 
    is_dtype_float=False,
    return_raw_data=False):
  ''' get a frame in 10bit YUV 444 format from a sequence file. 
      If W and H is not given, it's inferred from the sequeence file name

      Params
      ----------
        fname: file name of the sequence. It may contains <width>x<height> in the name
        frame_idx: the frame to be extracted, starting from 0
        W, H: width and height. If not given, it's inferred from the file name
        chroma_format: '420', '444', default '420' #'rgb', 'rgbp', 'yuv420p', 'yuv420p_10b'
        bit_depth: 8, 10, default 8
        is_dtype_flaot: if true, return float tensor with values in range [-1, 1]
        return_raw_data: if true, retype byte array

      Return
      ------
        Image in format HW3, and in range [-1, 1] if is_dtype_float is set. Otherwise, return data in uint16
  '''
  
  dtype = 'uint8'
  scale = 255
  if chroma_format == '444': 
    frame_length = W*H*3
  else: 
    frame_length = W*H*3//2
  if bit_depth == 10:
    frame_length *= 2 
    dtype = 'uint16'
    scale = 1024

  with open(fname, 'rb') as f:
    f.seek(start_pos)
    data = f.read(frame_length)

  if return_raw_data: return data, frame_length

  frame_data = np.frombuffer(data, dtype=dtype)

  if chroma_format=='444':
    # rgb planar
    frame=frame_data.reshape(3,H,W)
    frame = frame.transpose(1, 2, 0)
  else: 
    # yuv420 planar
    y = frame_data[:H*W].reshape(H, W)
    uv_length = H*W//4
    u = frame_data[H*W:H*W+uv_length].reshape(H//2, W//2)
    v = frame_data[H*W+uv_length:].reshape(H//2, W//2)
    # upsample u an v by kronecker product
    kernel = np.array([[1,1], [1,1]], dtype=dtype)

    u = np.kron(u, kernel)
    v = np.kron(v, kernel)
    img_yuv = np.stack([y, u, v])  # 3HW
    frame = img_yuv.transpose(1,2,0) # HW3

  if is_dtype_float: 
    frame = frame.astype(float) / scale * 2 - 1
  elif bit_depth==8:
    frame = frame.astype('uint16')
    frame *= 4

  return frame, frame_length

def get_filename(item, input_fname):
    filename = input_fname.split(os.path.sep)[-1].split('.yuv')[0]
    if 'qp' in filename:
        return filename
    filename = item.args.output_recon_fname.split(os.path.sep)[-1]
    if 'qp' in filename:
        return filename
    return item.args.output_recon_fname.split(os.path.sep)[-1] + '_' + item.args.output_recon_fname.split(os.path.sep)[-2]