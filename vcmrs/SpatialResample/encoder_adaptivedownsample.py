# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os, sys
import cv2
import vcmrs
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
import re

import numpy as np
import pandas as pd
import math
import torch
# import warnings # disable warnings

from pathlib import Path
import vcmrs.SpatialResample.models as spatial_model
from vcmrs.SpatialResample.models import scale_factor_generator 
from vcmrs.Utils.io_utils import enforce_symlink, makedirs

class AdaptiveDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
    
    vcmrs.log("Spatial Resampling : ===================== Calculate adaptive scaling factors ====================")
    
    self.generator = scale_factor_generator.AdaptiveScaleFactorGenerator(item, input_fname)
    all_scale_list = None
    
    if item.args.SpatialDescriptorMode in ['UsingDescriptor', 'GeneratingDescriptor', 'GeneratingDescriptorExit']:
      if (descriptor_file := item.args.SpatialDescriptor) is None:
        descriptor_file = os.path.join(os.path.dirname(os.path.dirname(vcmrs.__file__)), "Data", "spatial_descriptors",  os.path.basename(item.args.output_recon_fname) + ".csv")
        descriptor_file = re.sub(r'_qp\d+', '', descriptor_file)
            
      descriptor_dir = os.path.dirname(descriptor_file)
    

    if item.args.SpatialDescriptorMode in ['NoDescriptor', 'GeneratingDescriptor', 'GeneratingDescriptorExit']:  # generate?
      object_info = self.generator.get_object_information(item, input_fname)
    
    if item.args.SpatialDescriptorMode in ['GeneratingDescriptor', 'GeneratingDescriptorExit']: # save
      
      vcmrs.log("Spatial Resampling : start to write the csv file")
      vcmrs.log(f"Spatial Resampling : write the {descriptor_file}")
      
      if os.path.exists(descriptor_file) :
        vcmrs.error("Spatial Resampling : Error : descriptor_file already exist. Failed to generate the descriptor file. ")
        sys.exit(-1)
      
      makedirs(descriptor_dir)
      
      object_info.to_csv(descriptor_file, header=True, mode='w', sep=',', index=False)

      vcmrs.log("Spatial Resampling : Done : Generating the spatial descriptor is completed.")

      if item.args.SpatialDescriptorMode=='GeneratingDescriptorExit':
        vcmrs.log(f"Spatial Resampling : Exitting after saving")
        vcmrs.log(f"Encoding completed in 0 seconds")
        sys.exit(0)
      
    if item.args.SpatialDescriptorMode=='UsingDescriptor': # load
      if not os.path.exists(descriptor_file):
        vcmrs.error("Spatial Resampling : Error : descriptor_path does not exist.")
        sys.exit(-1)
            
      object_info = self.generator.read_descriptor(descriptor_file)

    all_scale_list = self.generator.generate_scale_factor_list(item, object_info)
    # optimal_scale_factor_list = generate_scale_factor_list(item, pre_analyzed_data)

    enforce_symlink(input_fname, output_fname)
    
    vcm_spatial_resampling_flag = item.args.SpatialResamplingFlag
    spatial_resample_width = item.args.SourceWidth    
    spatial_resample_height = item.args.SourceHeight
    spatial_resample_filter_idx = 0  # m70183; One-bit flag to select non-inner codec RPR filter and the ‘reserved’ filter.

    self._set_parameter(item, vcm_spatial_resampling_flag, spatial_resample_width, spatial_resample_height, spatial_resample_filter_idx)
    vcmrs.log("Spatial Resampling : ================================= complete  =================================")
    return 0

  def _set_parameter(self, item, vcm_spatial_resampling_flag, spatial_resample_width, spatial_resample_height, spatial_resample_filter_idx=0) :
    # sequence level parameter
    if vcm_spatial_resampling_flag == 0 :
      param_data = bytearray(1)
      param_data[0] = vcm_spatial_resampling_flag  
    else : 
      # sc_len = len(sc_list) : remove unnessary code 
      param_data = bytearray(6)
      param_data[0] = vcm_spatial_resampling_flag
      param_data[1] = (spatial_resample_width >> 8) & 0xFF 
      param_data[2] = spatial_resample_width & 0xFF
      param_data[3] = (spatial_resample_height >> 8) & 0xFF 
      param_data[4] = spatial_resample_height & 0xFF
      param_data[5] = spatial_resample_filter_idx
    item.add_parameter('SpatialResample', param_data=param_data)
    