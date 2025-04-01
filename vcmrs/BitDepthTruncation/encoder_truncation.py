
    
# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import datetime
import shutil
import vcmrs
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink
import numpy as np



class truncation(Component):
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
      vcmrs.log("========================================== truncation start =====================================================")
      if item._is_yuv_video:
        height = item.args.SourceHeight
        width = item.args.SourceWidth
        bytes_per_pixel = 2 if item.args.InputBitDepth > 8 else 1

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
        output_file = open(output_fname, 'wb')

        if item.args.FrameSkip>0:
          input_file.seek( item.args.FrameSkip * bytes_per_pixel * (y_size+uv_size*2) )
  
        while True:
          # read Y, U, V
          y_buffer = input_file.read(y_size * bytes_per_pixel)
          if len(y_buffer) == 0:  # check if at the end of the file
              break
          y = np.frombuffer(y_buffer, dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape((height, width))  
          u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)
          v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)

          y = y.copy()
          # right shift 1
          y = np.right_shift(y, 1) 

          output_file.write(y.astype(np.uint16 if bytes_per_pixel == 2 else np.uint8).tobytes())
          output_file.write(u.tobytes())
          output_file.write(v.tobytes())
          
        item.OriginalFrameIndices = item.OriginalFrameIndices[item.args.FrameSkip:]
        item.args.FrameSkip = 0
        
        # close file
        input_file.close()
        output_file.close()
        bit_depth_shift_flag = 1 if item.args.OriginalSourceHeight < item.args.BitTruncationRestorationHeightThreshold and item.args.OriginalSourceWidth < item.args.BitTruncationRestorationWidthThreshold else 0
        # bit_depth_shift_flag=1 means: shift left (restore) at the decoder
        
        bit_depth_shift_luma = 1 # always applied, so always signalled, as per adopted m71663 (bit depth shift cleanup).
        bit_depth_shift_chroma = 0
        bit_depth_luma_enhance = 1 if item.args.BitTruncationL

        # # post filtering flag assignment # s.kwak
        # if (item.args.OriginalSourceHeight < item.args.BitTruncationRestorationHeightThreshold and item.args.OriginalSourceWidth < item.args.BitTruncationRestorationWidthThreshold) # resolution condition
        # and (bit_depth_shift_flag==1) and (bit_depth_shift_luma==1):
        #   post_filtering_enable_flag = 1
        # else:
        #   post_filtering_enable_flag = 0 
        # 나중에 하자.       

        self._set_parameter(item, bit_depth_shift_flag, bit_depth_shift_luma, bit_depth_shift_chroma, bit_depth_luma_enhance)
        vcmrs.log("================================= complete truncation =================================")
        if item.args.TemporalResamplingPostHintFlag:
          self.update_post_hint_score(item, output_fname, height, width, item.args.InputChromaFormat, item.args.InputBitDepth, q_bit_depth=7, call_position='enc', path=None)
      #non-yuv files
      else:
        enforce_symlink(input_fname, output_fname)

  def _set_parameter(self, item, bit_depth_shift_flag, bit_depth_shift_luma = 0, bit_depth_shift_chroma = 0, bit_depth_luma_enhance=0):
    # sequence level parameter
    # default scale: 1 byte, framestoberestore: 2 bytes
    param_data = bytearray(4)
    param_data[0] = bit_depth_shift_flag
    param_data[1] = bit_depth_shift_luma
    param_data[2] = bit_depth_shift_chroma
    param_data[3] = bit_depth_luma_enhance
    # param_data[4] = post_filtering_enable_flag # 
    item.add_parameter('BitDepthTruncation', param_data=param_data)
    pass

  def update_post_hint_score(self,item, output_fname, height, width, InputChromaFormat, InputBitDepth, q_bit_depth=7, call_position='enc', path=None):
    from vcmrs.TemporalResample import resample_score
    import torch           
    infile = item.trd_enc_fname
    num_pivot = 0
    flag_line = ''
    target_idx = -1
    if os.path.exists(infile):
      with open(infile, 'r+') as f:
        lines = f.readlines()

      for i in range(len(lines)):
        if ":" in lines[i]:
          key, value = lines[i].split(':')
          key = key.strip()
          value = value.strip()

          if key in ["trph_quality_valid_flag"]:
            flag_line = lines[i]
            target_idx = i
      num_pivot = item.args.FramesToBeEncoded
      prefix, numbers = flag_line.split(":")
      numbers = numbers.strip()
      num_list = numbers.split(",")[:num_pivot]
      new_flag_line = f"{prefix}:{','.join(num_list)}\n"
      lines[target_idx] = new_flag_line
      with torch.no_grad():
        temporalassesser = resample_score._TemporalAssesser_(height, width)
        frame_scores_list = temporalassesser.process_yuvs_assessment(output_fname, InputChromaFormat, InputBitDepth, q_bit_depth=7, frame_num=num_pivot, call_position='enc', path=None)
        score_line = f"trph_quality_value:{','.join(str(int(flag)) for flag in frame_scores_list)}\n"
        with open(infile, 'w') as f:
            f.writelines(lines)
            f.write(score_line)        