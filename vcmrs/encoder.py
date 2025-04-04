# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import os
import time
import atexit
import shutil
import pprint
import cv2

import vcmrs

# testing

from vcmrs.Utils import component_utils
from vcmrs.Utils import encoder_opts
from vcmrs.Utils import io_utils
from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.Utils.codec_context import CodecContext
from vcmrs.InnerCodec import encoder_nnvvc
from vcmrs.InnerCodec.Utils import video_utils

import tempfile

# global variable
_ctx = None

def initialize(args):
  r"""
  Initialize the encoding environment.

  Args:
    args: input arguments
  """
  if not args.working_dir:
    args.working_dir = tempfile.mkdtemp(dir=args.working_dir)

  # make context  
  ctx = CodecContext(args)
  global _ctx
  _ctx = ctx

  # parse input files
  ctx.input_files = io_utils.get_input_files(args)

  # register exit handler to terminate nnmanager
  atexit.register(exit_handler)

  # make torch working in deterministic mode
  utils.fix_random_seed()
  utils.make_deterministic()
 
  return ctx

def exit_handler():
  global _ctx

  # inner codec clean
  encoder_nnvvc.exit_handler(_ctx)

  # clean working directory
  if not _ctx.input_args.debug:
    for item in _ctx.input_files:
      shutil.rmtree(item.working_dir)
  
def encode(args):
  # s_total_time = time.time()
  # initialize
  ctx = initialize(args)

  component_table = {
    'F': 'FormatAdapter',
    'T': 'TemporalResample',
    'C': 'Colorize',
    'S': 'SpatialResample',
    'R': 'ROI',
    'B': 'BitDepthTruncation',
    'J': 'JointFilter',
    'P': 'PostFilter',
    'D': 'Decolorize',
  }
  print(args.ComponentOrder)

  assert args.ComponentOrder
  pre_components_str, post_components_str = args.ComponentOrder.split('_')
  pre_components_str = 'F' + pre_components_str
  post_components_str = post_components_str + 'F'
  pre_components = [component_table[m] for m in pre_components_str]
  post_components = [component_table[m] for m in post_components_str]

  print(pre_components)
  print(post_components)
  
  
  # pre-inner components 
  # print('encoder, truncation')
  # pre_components = []
  # pre_components += ['FormatAdapter']
  # pre_components += ['TemporalResample']
  # pre_components += ['Colorize']
  # pre_components += ['SpatialResample']
  # pre_components += ['ROI']
  # pre_components += ['BitDepthTruncation']
  # pre_components += ['Decolorize']
  for item in ctx.input_files:
    s_time = time.time()
    vcmrs.log(f'Pre-inner processing file: {item.fname}')
    
    item.FrameRateRelativeVsInput = 1.0
    item.IntraPeriod = int(item.args.IntraPeriod)
    # find original resolution if not given as command-line arg
    if item.args.SourceWidth is None or item.args.SourceHeight is None:
      iname = os.path.abspath(item.fname)
      
      if item._is_yuv_video:
        pass
      elif item._is_dir_video:
        image_file_list = data_utils.get_image_file_list(iname)
        if len(image_file_list)>0:
          item.args.SourceHeight, item.args.SourceWidth, C = data_utils.get_img_resolution(os.path.join(iname,image_file_list[0]))
      elif os.path.isfile(iname) and data_utils.is_file_image(iname):
        item.args.SourceHeight, item.args.SourceWidth, C = data_utils.get_img_resolution(iname)

    # store original resolution so that it can be used for control in plugins, e.g. bit-truncation
    try:
      item.args.OriginalSourceWidth  = int(item.args.SourceWidth)
      item.args.OriginalSourceHeight = int(item.args.SourceHeight)
    except:
      item.args.OriginalSourceWidth  = None
      item.args.OriginalSourceHeight = None
    
    if item.args.FramesToBeEncoded<=0:
      iname = os.path.abspath(item.fname)
      video_info = video_utils.get_video_info(iname, item.args)
      item.args.FramesToBeEncoded  = video_info.num_frames - item.args.FrameSkip
      
    item.OriginalFrameIndices = [ f for f in range(item.args.FramesToBeEncoded+item.args.FrameSkip) ]
    
    in_fname = item.fname
    item.trd_enc_fname = os.path.abspath(os.path.join( item.working_dir, "TEMPAR.CFG"))
    item.trd_dec_fname = os.path.abspath(os.path.join( item.working_dir, "TEMPAR_OUT.CFG"))
    for c_name in pre_components:
      component_method = getattr(ctx.input_args, c_name)
      component = component_utils.load_component(c_name, "encoder", component_method, ctx)
      out_fname = item.get_stage_output_fname(f"pre_{c_name}")
      item.fname = os.path.abspath(in_fname)
      encoder_component_time_start = time.time() #component coding time
      component.process(os.path.abspath(in_fname), os.path.abspath(out_fname), item, ctx)
      encoder_component_time_duration = time.time() - encoder_component_time_start #component coding time
      vcmrs.log(f"[{c_name} at encoder done]. Time = {encoder_component_time_duration:.6}(s)") #component coding time
      in_fname = out_fname
      component = None # free memory after each iteration
      
    item.inner_in_fname = in_fname
    item.inner_out_fname = item.get_stage_output_fname(f"inner")
    vcmrs.log(f"[{os.path.basename(item.args.working_dir)}] Pre-inner processing done. Time = {(time.time() - s_time):.6}(s)")

  # inner codec
  s_time = time.time()
  encoder_nnvvc.process(ctx.input_files, ctx)
  vcmrs.log(f"[{os.path.basename(item.args.working_dir)}] Inner encoding done. Time = {(time.time() - s_time):.6}(s)")

  # post_components = []
  # post_components += ['SpatialResample']
  # post_components += ['ROI']
  # post_components += ['Colorize']
  # post_components += ['TemporalResample']
  # post_components += ['PostFilter']
  # post_components += ['BitDepthTruncation']
  # post_components += ['FormatAdapter']
  
  for item in ctx.input_files:
    s_time = time.time()
    vcmrs.log(f'Post-inner processing file: {item.fname}')
    
    item.FrameRateRelativeVsInnerCodec = 1.0

    in_fname = item.inner_out_fname
    for c_name in post_components:
      component_method = getattr(ctx.input_args, c_name)
      component = component_utils.load_component(c_name, "decoder", component_method, ctx)
      out_fname = item.get_stage_output_fname(f"post_{c_name}")
      item.fname = os.path.abspath(in_fname)
      decoder_component_time_start = time.time() #component coding time
      component.process(os.path.abspath(in_fname), os.path.abspath(out_fname), item, ctx)
      decoder_component_time_duration = time.time() - decoder_component_time_start #component coding time
      vcmrs.log(f"[{c_name} at decoder done]. Time = {decoder_component_time_duration:.6}(s)") #component coding time
      in_fname = out_fname
      component = None # free memory after each iteration
   
    # generate ouptut recon
    io_utils.gen_output_recon(in_fname, item)
    vcmrs.log(f"[{os.path.basename(item.args.working_dir)}] Post-inner processing done. Time = {(time.time() - s_time):.6}(s)")
  # vcmrs.log(f"[{os.path.basename(item.args.working_dir)}] Total encoding done. Time = {(time.time() - s_total_time):.6}(s)")

def main():
  r"""
  Main function.
  """
  args = encoder_opts.get_encoder_arguments()
  vcmrs.setup_logger("main_encoder", args.logfile, args.debug)
  if args.debug_source_checksum:
    checksums = utils.get_project_checksums()
    vcmrs.log("Encoder checksums: \n" + "\n".join(checksums))

    cpp_checksums_file = os.path.join( os.path.dirname( os.path.abspath(vcmrs.__file__) ), "cpp_sources.chk")  
    vcmrs.log(f"Checking C++ checksums: {cpp_checksums_file}")
    if os.path.isfile(cpp_checksums_file):
      if utils.test_checksums(cpp_checksums_file):
        vcmrs.log("C++ checksums MISMATCH")
        exit()
      else:
        vcmrs.log("C++ checksums OK")
    else:
      vcmrs.log("C++ checksums not found")

  vcmrs.log("Start encoding...")
  vcmrs.log('Input arguments: ')
  s = pprint.pformat(args.__dict__)
  vcmrs.log(s)

  cv2.setNumThreads(0) # disables spawning of many threads for load/write operations


  # start_time 
  start_time = time.time()

  encode(args)

  elapse = time.time()-start_time
  vcmrs.log(f"Encoding completed in {elapse} seconds")


if __name__ == '__main__':
  main()
