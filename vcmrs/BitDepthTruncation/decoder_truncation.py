
    
# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import datetime
import shutil
import vcmrs
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
import numpy as np
from vcmrs.BitDepthTruncation import clahe

class truncation(Component):
  def __init__(self, ctx):
    super().__init__(ctx)

  def apply_clahe(self, y, input_bit_depth, dtype_depth, dtype, clip_limit=2.0, tile_grid_size=(8, 8)):
    y = np.clip(y, 0, 2**input_bit_depth)
    
    scale_factor = (2**dtype_depth-1) // (2**input_bit_depth-1)
    y_scaled = y * scale_factor
    
    y_clahe = clahe.clahe_custom(y_scaled, clip_limit, tile_grid_size, dtype)
    
    y_result = (y_clahe // scale_factor).astype(dtype)
    return y_result
  

  def process(self, input_fname, output_fname, item, ctx):
      bit_depth_shift_flag, bit_depth_shift_luma, bit_depth_shift_chroma, bit_depth_luma_enhance = self._get_parameter(item)
      
      item.args.InputChromaFormat = "420"
      item.args.InputBitDepth = 10
      
      makedirs(os.path.dirname(output_fname))
        
      #if item._is_yuv_video:
      if (item._is_yuv_video):
        vcmrs.log("==========================================================decoder truncation process start!==================================================================================")
        vcmrs.log(f"bit_depth_shift_flag is: {bit_depth_shift_flag}")
        vcmrs.log(f"bit_depth_shift_luma is: {bit_depth_shift_luma}")
        vcmrs.log(f"bit_depth_shift_chroma is: {bit_depth_shift_chroma}")
        vcmrs.log(f"bit_depth_luma_enhance is: {bit_depth_luma_enhance}")

        height,width,C = item.video_info.resolution
        bytes_per_pixel = 2 if item.args.InputBitDepth > 8 else 1
        dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
        dtype_depth = 16 if bytes_per_pixel == 2 else 8

        input_bit_depth = item.args.InputBitDepth - bit_depth_shift_luma

        # Determine sizes based on format
        if item.args.InputChromaFormat == "420":
            y_size = width * height
            uv_size = (width // 2) * (height // 2)
            uv_shape = (height // 2, width // 2)
        elif item.args.InputChromaFormat == "422":
            y_size = width * height
            uv_size = (width // 2) * height
            uv_shape = (height, width // 2)
        elif item.args.InputChromaFormat == "444":
            y_size = width * height
            uv_size = width * height
            uv_shape = (height, width)
        else:
            raise ValueError("Unsupported chroma format: {}".format(item.args.InputChromaFormat))

        input_file = open(input_fname, 'rb')
        makedirs(os.path.dirname(output_fname))
        output_file = open(output_fname, 'wb')

        while True:
          # read Y, U, V
          y_buffer = input_file.read(y_size * bytes_per_pixel)
          if len(y_buffer) == 0:  # check if at the end of the file
              break
          y = np.frombuffer(y_buffer, dtype = dtype).reshape((height, width))  
          u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)
          v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape)

          #if bit_depth_shift_flag: 
          if (bit_depth_shift_flag) and (item.args.post_filtering_enable_flag==0): # modified (s.kwak)
            if bit_depth_shift_luma:
              vcmrs.log(f"bit_depth_shift_luma is applied: {bit_depth_shift_luma}")
              y = y.copy()
              # right shift 1
              y = np.left_shift(y, bit_depth_shift_luma) 

              if bit_depth_luma_enhance:
                y = self.apply_clahe(y, input_bit_depth, dtype_depth, dtype)
              output_file.write(y.astype(dtype).tobytes())

            else:
              if bit_depth_luma_enhance:
                y = self.apply_clahe(y, input_bit_depth, dtype_depth, dtype)
              output_file.write(y.astype(dtype).tobytes())
            
            if bit_depth_shift_chroma:
              u = u.copy()
              v = v.copy()
              u = np.left_shift(u, bit_depth_shift_chroma) 
              v = np.left_shift(v, bit_depth_shift_chroma) 
              output_file.write(u.astype(dtype).tobytes())
              output_file.write(v.astype(dtype).tobytes())
            else:            
              output_file.write(u.tobytes())
              output_file.write(v.tobytes())
        
          else:
              vcmrs.log("================================= Bit depth shift is skipped (Post-filter is applied) =================================")
              if bit_depth_luma_enhance:
                y = y.copy()
                y = self.apply_clahe(y, input_bit_depth, dtype_depth, dtype)
                output_file.write(y.astype(dtype).tobytes())
              else:
                output_file.write(y.tobytes())

              output_file.write(u.tobytes())
              output_file.write(v.tobytes())
            
        input_file.close()
        output_file.close()
        vcmrs.log("================================= complete truncation =================================")
      else:
        vcmrs.log("================================= Bit depth shift is skipped (Post-filter) =================================")
        enforce_symlink(input_fname, output_fname)
      
  def _get_parameter(self, item):
    return bit_depth_trunation_get_parameter(item)


def bit_depth_trunation_get_parameter(item): # moved here so that other tools can easily access this information as per adopted m71663 (bit depth shift cleanup).
    # sequence level parameter
    bit_depth_shift_flag = 0
    bit_depth_shift_luma = 0
    bit_depth_shift_chroma = 0
    bit_depth_luma_enhance = 0
    param_data = item.get_parameter('BitDepthTruncation')
    if param_data is not None:
      assert len(param_data) == 4, f'received parameter data is not correct: {param_data}'
      bit_depth_shift_flag = param_data[0]
      bit_depth_shift_luma = param_data[1]
      bit_depth_shift_chroma = param_data[2]
      bit_depth_luma_enhance = param_data[3]
    return bit_depth_shift_flag, bit_depth_shift_luma, bit_depth_shift_chroma, bit_depth_luma_enhance


