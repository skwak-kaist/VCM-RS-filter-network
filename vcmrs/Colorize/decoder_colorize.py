# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import vcmrs
import shutil
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.Colorize.colorizer_colorful import *
from vcmrs.Colorize.col_utils import *

from vcmrs.BitDepthTruncation.decoder_truncation import bit_depth_trunation_get_parameter
# usa luma shift from BitDepthTruncation, as allowed per adopted  m71663 (bit depth shift cleanup).

# component base class
class colorize(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    #self.log = print
    self.log = vcmrs.log

    self.colorizers = ColorizerColorful( )

  def process(self, input_fname, output_fname, item, ctx):

    vcmrs.log('######decoder colorize process###########')
    
    self._get_parameter(item)
    
    makedirs(os.path.dirname(output_fname))

    if item._is_dir_video: # input format is directory with separate files
    
      if os.path.exists(output_fname):
        os.remove(output_fname)
      enforce_symlink(input_fname, output_fname)
      #makedirs(output_fname)
      #self.colorizer.process_directory(input_fname, output_fname)
      
    elif item._is_yuv_video and (self.colorization_decisions is None):
    
      vcmrs.log('######decoder colorizer disabled###########')
      if os.path.exists(output_fname):
        os.remove(output_fname)
      enforce_symlink(input_fname, output_fname)
      
    elif item._is_yuv_video: # input format is yuv file,  convert to pngs and back
      
      item.args.InputChromaFormat = "420"
      item.args.InputBitDepth = 10
      
      H,W,C = item.video_info.resolution
      num_frames = 0

      shifts = []
      decisions = []
      for i in range(len(self.colorization_decisions)*self.colorization_period):
        idd = i//self.colorization_period # intra period idx
        shifts.append( self.luma_pre_shifts[idd] )
        decisions.append( self.colorization_decisions[idd] )

      self.colorizers.process_video_yuv(-1, input_fname, output_fname, W, H, item.args.InputChromaFormat, item.args.InputBitDepth, shifts, decisions)

    elif os.path.isfile(input_fname): 
    
      if os.path.exists(output_fname):
        os.remove(output_fname)
      makedirs(os.path.dirname(output_fname))
      enforce_symlink(input_fname, output_fname)
      
    else:
    
      assert False, f"Input file {input_fname} is not found"

  def _get_parameter(self, item):
    # usa luma shift from BitDepthTruncation, as allowed per adopted  m71663 (bit depth shift cleanup).
    bit_depth_shift_flag, bit_depth_shift_luma, bit_depth_shift_chroma, bit_depth_luma_enhance = bit_depth_trunation_get_parameter(item)
    lps = bit_depth_shift_luma
    
    self.colorization_decisions = None
    self.luma_pre_shifts = None
    
    bytes_data = item.get_parameter('Colorize')
    if bytes_data is None: return
   
    self.colorization_decisions = []
    self.luma_pre_shifts = []
    
    pos = 0
    
    perd = bytes_data[pos] | (bytes_data[pos+1]<<8)
    pos += 2

    vcmrs.log(f"VCMRS Colorizer decoding: period: {perd}")

    leng = bytes_data[pos] | (bytes_data[pos+1]<<8)
    pos += 2
    
    vcmrs.log(f"VCMRS Colorizer decoding: num: {leng}")

    self.colorization_period = perd
    
    for i in range(leng):
      ena = bytes_data[pos]
      pos += 1
      if ena:
        cidx = bytes_data[pos]
        pos += 1
        #lps = bytes_data[pos] # this is not transmitted now, due to adopted m71663 (bit depth shift cleanup).
        #pos += 1
      else:
        cidx = 255
        #lps = 0
    
      self.colorization_decisions.append( cidx )
      self.luma_pre_shifts.append( lps )
    