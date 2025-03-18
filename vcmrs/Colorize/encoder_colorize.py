# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import vcmrs
import shutil
import numpy as np
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.Colorize.colorizer_colorful import *
from vcmrs.Colorize.col_utils import *
from vcmrs.ROI.encoder_roi_generation import get_roi_accumulation_periods

# component base class
class colorize(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    #self.log = print
    self.log = vcmrs.log
    
    self.colorizers = ColorizerColorful()
      
  def process(self, input_fname, output_fname, item, ctx):
   
    vcmrs.log('######encoder colorize process###########')

    colorize_descriptor_mode = ctx.input_args.ColorizeDescriptorMode
    colorize_decision = ctx.input_args.ColorizeDecision
    colorize_descriptor_file = ctx.input_args.ColorizeDescriptorFile
    
    if colorize_decision=="on":
      colorize_decision = "dynamic" # switch here to use first" mode
      
    if item._is_dir_video: # input format is directory with separate files
    
      if os.path.exists(output_fname):
        os.remove(output_fname)    
      enforce_symlink(input_fname, output_fname)
      #makedirs(output_fname)
      #self.colorizer.process_directory(input_fname, output_fname)
      
    elif (item._is_yuv_video) and (colorize_decision=="off") and not (colorize_descriptor_mode=="saveexit"):
    
      if os.path.exists(output_fname):
        os.remove(output_fname)
      enforce_symlink(input_fname, output_fname)
      vcmrs.log( f"Colorizer off" )
      
    elif item._is_yuv_video: # input format is yuv file,  convert to pngs and back

      H = item.args.SourceHeight
      W = item.args.SourceWidth
      num_frames = item.args.FramesToBeEncoded
      
      if item.args.Configuration == 'AllIntra':
        num_frames_decision = -1 # all frames
        item.colorization_period = 1
      elif item.args.Configuration == 'LowDelay':
        num_frames_decision = 1 # first frame only
        item.colorization_period = num_frames*10 # infinite
      else:                
        num_frames_decision = -1 # all frames        
        item.colorization_period = item.IntraPeriod
      
      # list of frames for which colorization decision is made
      decision_frames_list = [ x+item.args.FrameSkip for x in range(0, num_frames, item.colorization_period)]
      
      # list of frames for which descriptors need to be generated
      generate_frames_list = None
      
      if colorize_descriptor_mode in ["save", "saveexit"]:
        #generate all
        generate_frames_list = [x+item.args.FrameSkip for x in range(num_frames)]
      elif colorize_descriptor_mode=="onthefly":
        if colorize_decision=="first":
          generate_frames_list = [item.args.FrameSkip]
        if colorize_decision=="dynamic":
          generate_frames_list = decision_frames_list
      
      if generate_frames_list is not None:  
        vcmrs.log(f"Colorize: generating descriptors...")
        
        input_fname1 = input_fname+".org.yuv"
        yuv_select_frames(generate_frames_list, input_fname, input_fname1, W, H, item.args.InputChromaFormat, item.args.InputBitDepth)
        
        input_fname2 = input_fname+".gray.yuv"
        yuv_chroma_clear(-1, input_fname1, 0, input_fname2, W, H, item.args.InputChromaFormat, item.args.InputBitDepth, True)
      
        similarities_per_colorizer = []
        for col_idx in range(self.colorizers.num):
          col_name = self.colorizers.names[col_idx]
          
          output_fname2 = f"{output_fname}.colorized_{col_name}.yuv"
          self.colorizers.process_video_yuv(num_frames_decision, input_fname1, output_fname2, W, H, item.args.InputChromaFormat, item.args.InputBitDepth, 0, col_idx)
          
          sims = calculate_chroma_similarity(-1, input_fname1, output_fname2, W, H, item.args.InputChromaFormat, item.args.InputBitDepth)
          similarities_per_colorizer.append( sims )
          
          if not item.args.debug:
            os.remove(output_fname2)
       
        if not item.args.debug:
          os.remove(input_fname1)
          os.remove(input_fname2)
       
        gen_descriptors = {}
        for fi in range(len(generate_frames_list)):
          f = generate_frames_list[fi]
          fo = item.OriginalFrameIndices[f]
          
          best_colorization_sim = 0.7
          best_colorization_idx = 255 # off
          for col_idx in range(self.colorizers.num):
            sim = similarities_per_colorizer[col_idx][fi]
            if sim>=best_colorization_sim:
              best_colorization_idx = col_idx
          gen_descriptors[fo] = best_colorization_idx
          
          
        if colorize_descriptor_mode in ["save", "saveexit"]:
          vcmrs.log(f"Colorize: saving {colorize_descriptor_file}")
          save_descriptor_to_file(colorize_descriptor_file, gen_descriptors)
        
        if colorize_descriptor_mode=="saveexit":
          vcmrs.log(f"Colorize: exiting after saving")
          vcmrs.log(f"Encoding completed in 0 seconds")
          sys.exit(0)
          
      
      item.colorization_decisions = []
      
      if colorize_decision in ["first", "dynamic"]:
      
        if colorize_descriptor_mode=="load":
          vcmrs.log(f"Colorize: loading descriptor {colorize_descriptor_file}")
          gen_descriptors = load_descriptor_from_file(colorize_descriptor_file)
        
        decision_descriptors = []
        for f in decision_frames_list:
          fo = item.OriginalFrameIndices[f]
          item.colorization_decisions.append( gen_descriptors[fo] )
        
      else:
        # force particular colorizer or off
        if colorize_decision=="off":
          best_colorization_idx = 255 # off
        elif colorize_decision=="0":
          best_colorization_idx = 0
        elif colorize_decision=="1":
          best_colorization_idx = 1
        
        item.colorization_decisions = []
        for f in decision_frames_list:
          item.colorization_decisions.append(best_colorization_idx)

      enforce_symlink(input_fname, output_fname)

      # Produce output stream
      
      # VCMRS is able to output sequence-level synchronization altogether with all units.
      # Therefore the period of sending Colorizer data has to be NO longer than period of sending ROI updates for retargeting process.
      
      accumulation_period, roi_update_period =  get_roi_accumulation_periods(item.args, item.IntraPeriod, item.FrameRateRelativeVsInput)

      #vcmrs.log(f"Colorizer roi_update_period:{roi_update_period}, colorization_period:{item.colorization_period}")

      if roi_update_period < item.colorization_period:
      
        # duplicate (repeat) colorization decisions to for every frame where signallizationis is done (e.g .every frame in LowDelay e2e)
        
        vcmrs.log(f"Colorizer roi_update_period<colorization_period:   {roi_update_period}<{item.colorization_period}")
        dense_decisions = []
        for f in range(0, num_frames, roi_update_period):
          nf = f//item.colorization_period
          dense_decisions.append( item.colorization_decisions[nf] )
        item.colorization_decisions = dense_decisions
        item.colorization_period = roi_update_period
      
      # no luma_pre_shift due to adopted m71663 (bit depth shift cleanup).
      self._set_parameter(item)
      
    elif os.path.isfile(input_fname): 
    
      if os.path.exists(output_fname):
        os.remove(output_fname)
      makedirs(os.path.dirname(output_fname))
      enforce_symlink(input_fname, output_fname)
      
    else:
      assert False, f"Input file {input_fname} is not found"

  # no luma_pre_shift =due to adopted m71663 (bit depth shift cleanup).
  def _set_parameter(self, item):
    # sequence level parameters, different for each colorization_period

    result = []

    perd = item.colorization_period
    result.append( perd & 0xFF )
    result.append( (perd>>8) & 0xFF )
    vcmrs.log(f"VCMRS Colorizer encoding: colorization period: {perd}")
    
    leng = len(item.colorization_decisions) # number of colorization_periods for the knownledge of frame muxer
    result.append( leng & 0xFF )
    result.append( (leng>>8) & 0xFF )
    vcmrs.log(f"VCMRS Colorizer encoding: num: {leng}")
         
    for i in range(leng):
      en = 1 if item.colorization_decisions[i]<255 else 0
      result.append(en)
      if en:
        result.append(item.colorization_decisions[i])
        #result.append(luma_pre_shift)  # no luma_pre_shift due to adopted m71663 (bit depth shift cleanup).

      
    item.add_parameter('Colorize', param_data=bytearray(result) )

