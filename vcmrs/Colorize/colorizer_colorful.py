import os
import torch
import glob
import cv2
import numpy as np
from vcmrs.Colorize.colorful import siggraph17ycbcr as siggraph17ycbcr
from vcmrs.Colorize.colorful import eccv16ycbcr as eccv16ycbcr
from vcmrs.Colorize.colorful import util as util
from vcmrs.Colorize.col_utils import *

import random
import warnings
import time
import torch.onnx
from e2evc.Utils import ctx # use integer-convolutions like in TemporalResampling

# disable all non-deterministic behaviour
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

def setup_seed(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False    
        torch.backends.cudnn.deterministic = True
setup_seed()

class ColorizerColorful:

  def __init__(self):
    self.use_gpu = False

    # disable all non-deterministic behaviour
    setup_seed()
    #cpu_num = 1
    #os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    #os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    #os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    #os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    #os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    #torch.set_num_threads(cpu_num) 

    torch.use_deterministic_algorithms(True)
    #torch.set_num_interop_threads(1)

    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        # cancel dynamic algorithm
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    self.networks = []
    self.networks.append( eccv16ycbcr(pretrained=True).eval() ) 
    self.networks.append( siggraph17ycbcr(pretrained=True).eval() )
    
    self.names = ["eccv16ycbcr", "siggraph17ycbcr"]
    
    self.num = len(self.networks)

    if(self.use_gpu):
      for n in self.networks:
        n.cuda()
    
  @ctx.int_conv()  # use integer-convolutions like in TemporalResampling
  def process_video_yuv(self, num_frames, input_fname, output_fname, width, height, chroma_format, bit_depth, shift_lefts, colorizer_decisions):
    
    y_size, uv_size, uv_shape, bytes_per_pixel, frame_size_bytes = determine_format_characteristics(width, height, chroma_format, bit_depth)
    
    input_file = open(input_fname, 'rb')
    output_file = open(output_fname, 'wb')
    
    frame_index = 0
    while True:
      y_buffer = input_file.read(y_size * bytes_per_pixel)
      if len(y_buffer) == 0:
        break
      dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
      y = np.frombuffer(y_buffer,                                   dtype=dtype).reshape((height, width)).copy()
      u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape).copy()
      v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=dtype).reshape(uv_shape).copy()
  
      try:
        decision = colorizer_decisions[frame_index]
      except:
        decision = colorizer_decisions
         
      if decision<255:   
        try:
          sh = shift_lefts[frame_index]
        except:
          sh = shift_lefts
      
        ysh = np.left_shift(y, sh) 
    
        tens_y_rs = util.preprocess_img_yuv(ysh, bit_depth, HW=(256,256))
        if(self.use_gpu):
          tens_y_rs = tens_y_rs.cuda()

        with torch.inference_mode():
          with ctx.int_conv(): # use integer-convolutions like in TemporalResampling
            res = self.networks[decision](tens_y_rs).cpu()
        
        #img_bw  = util.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
        u, v  = util.postprocess_tens_yuv(u,v, bit_depth, res)
      
      output_file.write(y.tobytes())
      output_file.write(u.tobytes())
      output_file.write(v.tobytes())
      
      frame_index += 1
      if (num_frames>0) and (frame_index>=num_frames): break
      
      
    input_file.close()
    output_file.close()
  
  @ctx.int_conv()  # use integer-convolutions like in TemporalResampling
  def process_directory(self, inp_dir, out_dir, num_frames, colorizer_decisions):
  
    if not os.path.isdir(inp_dir):
      assert False, "Input file should be directory!"
    inp_fnames = sorted(glob.glob(os.path.join(inp_dir, '*.png')))
    
    seq_length = len(inp_fnames)
    if num_frames>=0: seq_length = min(seq_length, num_frames)
        
    for f in range(seq_length):
      inp_fname = inp_fnames[f]
      basename = os.path.basename(inp_fname)
      out_fname = os.path.join(out_dir, basename)
      
      img = cv2.imread(inp_fname) # BGR

      try:
        decision = colorizer_decisions[f]
      except:
        decision = colorizer_decisions
      
      if len(img.shape)>2: # remove 4-th component if exists in the file
        if img.shape[2]>3:
          img = img[:,:,0:3]
        
      assert img is not None, f"Cannot read file:{inp_fname}"
      
      if decision<255:

        img = img[:,:, ::-1] # BGR<->RGB
          
        (tens_l_orig, tens_l_rs) = util.preprocess_img(img, HW=(256,256))
        if(self.use_gpu):
          tens_l_rs = tens_l_rs.cuda()
        
        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel
        img_bw  = util.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

        with torch.inference_mode():
          with ctx.int_conv(): # use integer-convolutions like in TemporalResampling
            res = self.networks[decision](tens_l_rs).cpu()

        out_img = util.postprocess_tens(tens_l_orig, res)
        
        out_img = (out_img*255).astype(np.uint8)
        out_img = out_img[:,:, ::-1] # BGR<->RGB
      else:
        out_img = img

      cv2.imwrite(out_fname, out_img)
      
    return
