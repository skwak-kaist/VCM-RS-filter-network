# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import vcmrs
import shutil
import numpy as np
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.Colorize.col_utils import *

# component base class
class decolorize(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    #self.log = print
      
  def process(self, input_fname, output_fname, item, ctx):
    
    vcmrs.log('######encoder decolorize process###########')

    if ctx.input_args.ColorizeDecision=="off":
      enforce_symlink(input_fname, output_fname)
      vcmrs.log( f"Decolorizer off" )
    elif item._is_dir_video: # input format is directory with separate files           
      enforce_symlink(input_fname, output_fname)      
    elif item._is_yuv_video:

      H = item.args.SourceHeight
      W = item.args.SourceWidth
      num_frames = item.args.FramesToBeEncoded
      
      try:
        colorization_decisions = item.colorization_decisions
        colorization_period =  item.colorization_period
      except:
        colorization_decisions = None
        colorization_period = None
       
      if colorization_decisions is not None:
        graying_specification = []
        for i in range( len(colorization_decisions)*colorization_period ):
          graying_specification.append( colorization_decisions[i//colorization_period]<255 )
        vcmrs.log( f"Decolorizer enabled" )

        yuv_chroma_clear(num_frames, input_fname, item.args.FrameSkip, output_fname, W, H, item.args.InputChromaFormat, item.args.InputBitDepth, graying_specification)
        item.OriginalFrameIndices = item.OriginalFrameIndices[item.args.FrameSkip:]

      else:
        vcmrs.log( f"Decolorizer disabled" ) #org
        enforce_symlink(input_fname, output_fname)
      #exit()
      
    elif os.path.isfile(input_fname):
      if os.path.exists(output_fname):
        os.remove(output_fname)
      makedirs(os.path.dirname(output_fname))
      enforce_symlink(input_fname, output_fname)
    else:
      assert False, f"Input file {input_fname} is not found"
