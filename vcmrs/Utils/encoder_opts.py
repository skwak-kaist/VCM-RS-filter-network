# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import os
import copy
import argparse

def RoiRetargetingModeType(value):
  valid_modes = ["off", "first", "first+pad", "sequence", "sequence+pad", "dynamic+pad", "dynamic+fit"]
  value = value.lower()
  if value in valid_modes: 
    return value.split("+")
    
  is_pad = "+pad" in value
  # "+fit" does not make sense in other modes
  
  value = value.replace("+pad","")

  try:
    user_resolution = value.split("x")
    W = int(user_resolution[0])
    H = int(user_resolution[2])
    result = ["user", W,H]
    if is_pad:
      result += ["pad"]
    return result
  except:
    raise argparse.ArgumentTypeError(f'Invalid value: {value}')
 
  

def get_encoder_parser():

  parser = argparse.ArgumentParser(description='Encode video or images for machine consumption')

  # data I/O
  parser.add_argument('--output_dir', type=str, default='./output', 
                      help='Directory of the output data. Default: output.')
  parser.add_argument('--output_bitstream_fname', type=str, default=os.path.join('bitstream', '{bname}.bin'), 
                      help='Output bitstream name. Default empty string which means the output bitstream file name is same as the input image or video file name, in {output_dir}/bitstream directory.')
  parser.add_argument('--output_recon_fname', type=str, default=os.path.join('recon', '{bname}'), 
                      help='file name for the output reconstructed data. Default empty string which means the output recon file name is the same as the input image or video file name, in {output_dir}/recon directory.')
  parser.add_argument('--output_frame_format', type=str, default='frame_{frame_idx:06d}.png', 
                      help='The output frame file name format for video filese. data. Default: frame_{frame_idx:06d}.png, which will generage frames with file names like frame_000000.png, frame_000001.png, .... One can also use variable frame_idx1 for frame index starting from 1, for example {frame_idx1:06d}.png. ')

  parser.add_argument('--directory_as_video',  action='store_true',
                      help='If specified, input directory is treated as video with frames in png or jpg file format')

  parser.add_argument('--input_prefix', type=str, default='',
                      help='The prefix to be applied to the input files. Default is empty.')
  parser.add_argument('input_files', type=str, nargs='*', default='', 
                      help='''File name or directory name of the input data. Valid inputs are: image file in jpg or png format; video frames in a directory(with --directory_as_video argument); raw video data in YUV format with extention ".yuv"; list file with extention ".txt" or ".lst" that contains files to be encoded. If --input_dir is given, it is added to the input file as prefix. If the input file is a directory, the data in the directory will be processed either as images or frames of a video, depending on --directory_as_video argument.''')

  parser.add_argument('--working_dir', type=str, default=None,
                      help='working directory to store temporary files, defualt is system temporary directory')

  # Input data for YUV data
  parser.add_argument('--SourceWidth', type=int, default=None, 
                      help='The width of the frames in a video in YUV format. Default None.')
  parser.add_argument('--SourceHeight', type=int, default=None, 
                      help='The height of the frames in a video in YUV format. Default None.')
  parser.add_argument('--InputBitDepth', type=int, default=8, 
                      help='The input bit depth of the input video in YUV format, 8 or 10. Default 8')
  parser.add_argument('--InputChromaFormat', type=str, default='420', 
                      help='The chroma format of the input video in YUV format, 420 or 444. Default: 420')
  parser.add_argument('--InputYUVFullRange', action='store_true', 
                      help='If specified, the input YUV sequence is in full range')

  # Video data encoding configurations
  parser.add_argument('--FrameRate', type=int, default=30, 
                      help='The frame rate of the input video. Default 30.')
  parser.add_argument('--FrameSkip', type=int, default=0,
                      help='Number of frames skipped from the input')
  parser.add_argument('--FramesToBeEncoded', type=int, default=0, 
                      help='Number of frames to be encoded. Default 0: all frames are encoded')

  parser.add_argument('--IntraPeriod', type=int, default=32,
                      help='Intra period, as defined in VVC. Default: 32')

  parser.add_argument('--Configuration', type=str, default="RandomAccess",
                      help='GoP configuration, one of RandomAccess, LowDealy and AllIntra. Applies to video coding only')

  parser.add_argument('--quality', type=int, default=22,
                      help='Compression quality in range 0 - 100. Default: 22')

  parser.add_argument('--NNIntraQPOffset', type=int, default=-5, 
                      help='QP offset for NN intra model. Note that this argument has not effect for image coding')

  parser.add_argument('--IntraHumanAdapter', type=int, default=1, 
                      help='Apply IntraHumanAdapter. Default: 1')

  parser.add_argument('--InterMachineAdapter', type=int, default=1, 
                      help='Apply InterMachineAdapter. Default: 1')
                      
  parser.add_argument('--IntraFallbackQP', type=int, default=49, 
                      help='Fallback to CVC codec when intra QP is higher than IntraFallbackQP value')

  parser.add_argument('--ResolutionThreshold', type=int, default=1920, 
                      help='If the longest side of an input video is longer than this value, the input video is resized according to the OversizedVideoScaleFactor')

  parser.add_argument('--OversizedVideoScaleFactor', type=float, default=0.75, choices=[0.25, 0.5, 0.75], 
                      help='If an input video is rescaled, this scale factor is applied')


  # encoding tools configuration
  parser.add_argument('--FormatAdapter', type=str, default="formatadapter",
                      help='Input/output format adapter')

  parser.add_argument('--Colorize', type=str, default="colorize", 
                      help='Method for Colorize component')

  parser.add_argument('--ColorizeDecision', type=str, default="on", choices=['off','0','1','on'],
                      help='"off": turned off,   "0","1": manually select network, "on": automatically select and enable network')
                      
  parser.add_argument('--ColorizeDescriptorFile', type=str, default=None,
                      help='File name of ROI descriptor file to be loaded. Default value is None, indicating RoI is generated by analysis network specified by RoIGenerationNetwork')

  parser.add_argument('--ColorizeDescriptorMode', type=str, default="onthefly", choices=["load", "save", "saveexit", "onthefly"],
                      help='"load": load from ColorizeDescriptorFile and use it. "save","saveexit": generate all decisions and save to ColorizeDescriptorFile and optionally exit, "onthefly": generte decisions on-the-fly for the needed frames only. ')

  parser.add_argument('--Decolorize', type=str, default="decolorize", 
                      help='Method for Decolorize component')
  
  parser.add_argument('--BitDepthTruncation', type=str, default="truncation",
                      help='Method for bit depth truncation')

  parser.add_argument('--TemporalResample', type=str, default="resample",
                      help='Method for TemporalResample component')

  parser.add_argument('--TemporalScale', type=int, default=4, choices=[2,4,8,16],
                      help='Temporal resampling rate.')

  parser.add_argument('--TemporalResampleMethod', type=int, default=0, choices=[1,0],
                      help='Temporal resampling mode.0 represents interpolation, 1 represents extrapolation.')

  parser.add_argument('--TemporalExtrapolationResampleNum', type=int, default=2, choices=[2, 3, 4, 5],
                      help='The number of sampled frames in temporal extrapolation.')

  parser.add_argument('--TemporalExtrapolationPredictNum', type=int, default=2, choices=[1, 2, 3, 4],
                      help='The number of predicted frames in temporal extrapolation.')
  
  parser.add_argument('--TemporalResamplingPostHintFlag', type=int, default=0,
                      help='Method for temporal resampling post hint, default Bypass.')
  
  parser.add_argument('--PostHintSelectThreshold', type=int, default=6,choices=range(0,101),
                      help='Minimum alled ROI ratio for temporal resampling post hint') 
  
  parser.add_argument('--SpatialResample', type=str, default="AdaptiveDownsample",
                      help='Method for SpatialResample component')

  parser.add_argument('--ROI', type=str, default="roi_generation",
                      help='Method for ROI component')
  
  parser.add_argument('--NnlfSwitch', type=str, default="NnlfSliceBased", choices=['NnlfSliceBased','Bypass'],
                      help='NnlfSliceBased indicates NNLF usage is switched on slice level, Bypass indicates NNLF is turned off')
  
  parser.add_argument('--NnlfSwitchEnable', type=str, default="Disabled", choices=['Enabled','Disabled'],
                      help='Enabling of Nnlf switch, default Disabled. This can be overriden automatically by NnlfSwitch=NnlfSliceBased')
  
  parser.add_argument('--nnlf_roi_fallback_min', type=float, default=0.25,
                      help='Minimum of the ROI area divided by total pixels calculated solely from the descriptors under which fallbackmode is enabled')
  
  parser.add_argument('--nnlf_roi_fallback_max', type=float, default=1,
                      help='Maximum of the ROI area divided by total pixels calculated solely from the descriptors under which fallbackmode is enabled')
  
  parser.add_argument('--nnlf_roi_size_limit_RA', type=int, default=98000,
                      help='Minimum allowed ROI size for NN-loop filter for random access')
  
  parser.add_argument('--nnlf_roi_size_limit_LD', type=int, default=98000,
                      help='Minimum allowed ROI size for NN-loop filter for low delay')
  
  parser.add_argument('--nnlf_roi_size_limit_AI', type=int, default=73000,
                      help='Minimum allowed ROI size for NN-loop filter for all intra')

  parser.add_argument('--InnerCodec', type=str, default="NNVVC", choices=['NNVVC', 'VTM'],
                      help='Method for InnerCodec component, default NNVVC')

  parser.add_argument('--IntraCodec', type=str, default='LIC', 
                      help='Method for end-to-end intra codec')

  parser.add_argument('--PostFilter', type=str, default="Bypass",
                      help='Method for PostFilter component')

  parser.add_argument('--BitTruncationRestorationWidthThreshold', type=int, default=1920,
                      help='Limit of width of the original image, above which bit-truncation process is not restored at the decoder')

  parser.add_argument('--BitTruncationRestorationHeightThreshold', type=int, default=1080,
                      help='Limit of height of the original image, above which bit-truncation process is not restored at the decoder')
                      
  parser.add_argument('--RoIGenerationNetwork', type=str, default="faster_rcnn_X_101_32x8d_FPN_3x", choices=["faster_rcnn_X_101_32x8d_FPN_3x", "yolov3_1088x608"],
                      help='Network to be used for RoI generation')

  parser.add_argument('--RoIDescriptor', type=str, default=None,
                      help='File name of ROI descriptor file to be loaded. Default value is None, indicating RoI is generated by analysis network specified by RoIGenerationNetwork')

  parser.add_argument('--RoIDescriptorMode', type=str, default="load", choices=["load", "generate", "save", "saveexit"],
                      help='"load": load from RoIDescriptor and use it, if RoIDescriptor is None, generate. "generate": use RoIGenerationNetwork on-the-fly.   "save","saveexit": use RoIGenerationNetwork and save to disk and optionally exit')

  parser.add_argument('--RoIAccumulationPeriod', type=int, default=0,
                      help='If RoIAccumulationPeriod==0, it is treated as 1 in AI encoding, and as IntraPeriod otherwise. If>0: value is adjusted to temporal resampling. If<0: value is used directly')

  parser.add_argument('--RoIAccumulationWindowExtension', type=int, default=0,
                      help='Extension of Accumulation Period to the past (previous frames) of the sequence')

  parser.add_argument('--RoIBackgroundFilter', type=float, default=0.0,
                      help='Background filter cut-off frequency, normalized 0 to 1. Values<0.001 use backward-compatible low-pass filter.')

  parser.add_argument('--RoIRetargetingBackgroundScale', type=int, default=15, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], # roi_consts.MAXIMAL_SCALE_FACTOR
                      help='Default scale index [0-15] for the background. Value 15 means 0% means graying out background. Value 0 means 100% meaning preserving the background and all RoIs without scale change .')

  parser.add_argument('--RoIRetargetingMode', default="sequence", type=RoiRetargetingModeType, 
                      help='"off": retargeting is disabled, "sequence": decision about retargeted content resolution is based on the whole sequence, "first": decision about retargeted content resolution is based on first frames, "WIDTHxHEIGHT": use specific WIDTH and HEIGHT, e.g. 500x500. The mentioned modes can optionally use modifier "+pad": pad the retargeted content to the original sequence resolution. Other modes: "dynamic+pad": decision about retargeted content resolution is made dynamically on-the-fly, and then padded to the original resulution, "dynamic+fit": scale dynamically generated retargeted content to fit the original resolution.')

  parser.add_argument('--RoIRetargetingMaxNumRoIs', type=int, default=11, 
                      help='Maximal number of RoIs used for retargeting; when exceeded, a single bounding-box RoI is used')

  parser.add_argument('--RoIAdaptiveMarginDilation', type=int, default=0,
                      help='1: Enables adaptive dilation of ROI margins. ): Disabled')

  parser.add_argument('--RoIInflation', type=str, default="auto", 
                      help='How much extend RoIs size and how to dillute RoI scales: value, e.g. 10, or in percents "10percent", "10%"')

  parser.add_argument('--SpatialDescriptorMode', type=str, default="NoDescriptor", choices=['GeneratingDescriptor', 'GeneratingDescriptorExit', 'UsingDescriptor', 'NoDescriptor'],
                      help='"GeneratingDescriptor": save the spatial descriptor, "GeneratingDescriptorExit" ... and exit. "UsingDescriptor": run the spatial resampling using the spatial descriptor. "NoDescriptor": run the spatial resampling without generating nor using the spatial descriptor.')
  
  parser.add_argument('--SpatialDescriptor', type=str, default=None,
                      help='File name of spatial descriptor file to be saved and loaded. Default value is None, indicating the spatial descriptor is located at "<VCM-RS root>/Data/spatial_descriptors"')

  parser.add_argument('--SpatialResamplingFlag', type=int, default=1,
                      help='This flag indicates whether the decoded video, consisting of downscaled pictures, requires upscaling. The default value is zero, meaning that spatial upscaling is not performed in VCM.')
  # Remove deprecated code (bicubic placeholder)
  # parser.add_argument('--Spatial_scalefactor_dir', type=str, default=None, 
  #                     help='This debug mode can be used in the case where the SpatialResamplingFlag is set to 1. The input of this parameter is the directory of the scale list.')
  parser.add_argument('--VCMBitStructOn', type=int, default=1, 
                      help='1 for turning on the new VCM bitstream structure design and 0 for turning off')
  
  parser.add_argument('--BitTruncationLumaEnhancement', type=int, default=1, 
                    help='1 for turning on the luma enhancement and 0 for turning off')

  parser.add_argument('--JointFilter', type=str, default='filter', help='Joint filtering')
  parser.add_argument('--JointFilterPreModel', type=str, default='FilterV8')
  parser.add_argument('--JointFilterPostModel', type=str, default='FilterV8')
  parser.add_argument('--JointFilterSelectionAlgorithm', type=str, default='algo1')
  parser.add_argument('--JointFilterPatchSize', type=int, default=256)
  parser.add_argument('--JointFilterPreDomain', type=str, default='RGB', choices=['RGB', 'YUV'])
  parser.add_argument('--JointFilterPostDomain', type=str, default='RGB', choices=['RGB', 'YUV'])
  parser.add_argument('--ComponentOrder', type=str, default='TSRB_RSTB', help='Order of pre-/post-components')


  return parser


def get_system_parser(parser):
  # system configuration
  parser.add_argument('--logfile', default='', type=str,
                      help='Path of the file where the logs are saved. By default the logs are not saved.')
  parser.add_argument('--debug', action='store_true', 
                      help='In debug mode, more logs are printed, intermediate files are not removed')
  parser.add_argument('--debug_skip_vtm', action='store_true', 
                      help='In debug mode, skip the VTM encoding')
  parser.add_argument('--debug_source_checksum', action='store_true', 
                      help='Print md5 checksums of source files')
  parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for parallel computing, for processing video only')
  parser.add_argument('--cvc_dir', type=str, default=os.path.join(\
                        os.path.dirname(os.path.dirname(__file__)), 'InnerCodec', 'VTM'),
                      help='directory of the CVC encoder directory')
  parser.add_argument('--NNConfigs', type=str, default="run_config.yaml",
                      help='Intra codec config file name. Default: run_config.yaml')
  parser.add_argument('--single_chunk', action = 'store_true',
                      help = 'Encode entire sequence in a single chunk')
  parser.add_argument('--port', type=str, default='*',
                      help='Port number for internal communication. Default: *, means NNManager select a free port')
  parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                      help='Path to ffmpeg executable')

def get_encoder_arguments(args=None, system_args=None):
  parser = get_encoder_parser()
  if not system_args: 
    get_system_parser(parser)

  if not args: 
    args = parser.parse_args()
  else:
    args = parser.parse_args(args, copy.deepcopy(system_args))

  if args.TemporalScale == -1: args.TemporalScale = 4
  # for VTM inner codec
  if args.InnerCodec == 'VTM':
    args.IntraFallbackQP = -1
    args.InterMachineAdapter = 0

  # for All intra with NNVVC inner codec, set intra period to be 1
  if args.Configuration == 'AllIntra':
    args.IntraPeriod = 1
    args.NNIntraQPOffset = 0
  elif args.Configuration == 'LowDelay':
    args.IntraPeriod = 99999999 # large enough for any video

  # ROI accumulaion period adjustment - moved to internals of ROI plugin;  Here intra period is unknown due to temporal resampling
  if args.SpatialResample == "Bypass":
       args.SpatialResamplingFlag = 0

  print('********************************')
  print(f'args:{args}')
  return args

