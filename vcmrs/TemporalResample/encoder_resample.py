# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import datetime
import shutil
import vcmrs
from vcmrs.Utils.io_utils import enforce_symlink, makedirs
from vcmrs.Utils.component import Component
from vcmrs.InnerCodec.Utils import video_utils
from . import resample_data
from vcmrs.Utils import utils

def rm_file_dir(fname):
    if os.path.exists(fname):
        if os.path.isdir(fname):shutil.rmtree(fname)
        else:os.remove(fname)

#deprecated
def yuvTopngs(input_fname, item, pngdir, num_frames, video_info):
    makedirs(pngdir)
    start = item.args.FrameSkip
    end = start + num_frames - 1
    pixfmt = video_utils.get_ffmpeg_pix_fmt(video_info)
    tmpname = os.path.join(pngdir, "frame_%06d.png")
    cmd = [
      item.args.ffmpeg, '-y', '-nostats', '-hide_banner', '-loglevel', 'error',
      '-threads', '1',
      '-f', 'rawvideo',
      '-s', f'{item.args.SourceWidth}x{item.args.SourceHeight}',
      '-pix_fmt', pixfmt,
      '-i', input_fname,
      '-vf' ,f'select=between(n\,{start}\,{end})',
      '-vsync', '0',
      '-y',
      '-pix_fmt', 'rgb24', 
      tmpname] # %06d.png

    err = utils.start_process_expect_returncode_0(cmd, wait=True)
    assert err == 0, "convert yuv to png failed."

class resample(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    self.scale_factor_mapping = resample_data.scale_factor_id_mapping

  def _updateExtArgs(self, item, num_frames, mixTemporalScale):
      item.args.FrameSkip = 0
      item.args.FramesToBeEncoded = num_frames
      if item.IntraPeriod > 1:
          item.IntraPeriod = int(item.IntraPeriod / mixTemporalScale)
      item.FrameRateRelativeVsInput = item.FrameRateRelativeVsInput / mixTemporalScale
      vcmrs.log(f'item.args.FramesToBeEncoded:{item.args.FramesToBeEncoded}')

  def _ext_files(self, inputdir, outputdir, pic_all_index, item, video_info, yuvFile=False):
      if not os.path.isdir(inputdir):
          assert False, "Input file should be directory!"
      fnames = sorted(glob.glob(os.path.join(inputdir, '*.png')))
      rm_file_dir(outputdir)

      output_original_frame_indices = []
      for i, idx in enumerate(pic_all_index):
          idxfs = idx+item.args.FrameSkip
          vcmrs.debug(f'need frame idx:{idxfs}')
          fname = fnames[idxfs]
          makedirs(outputdir)
          output_frame_fname = os.path.join(outputdir, f"frame_{i:06d}.png")
          rm_file_dir(output_frame_fname)
          vcmrs.debug(f'{os.path.basename(fname)} ===>> {os.path.basename(output_frame_fname)}')
          enforce_symlink(fname, output_frame_fname)
          output_original_frame_indices.append( item.OriginalFrameIndices[idxfs] )

      if yuvFile:  # deprecated
          tmpdir = os.path.splitext(outputdir)[0]
          rm_file_dir(tmpdir)
          os.renames(outputdir, tmpdir)
          pixfmt = video_utils.get_ffmpeg_pix_fmt(video_info)
          out_frame_fnames = os.path.join(tmpdir, 'frame_%06d.png')
          cmd = [
              item.args.ffmpeg, '-y', '-nostats', '-hide_banner', '-loglevel', 'error',
              '-threads', '1',
              '-i', out_frame_fnames,
              '-f', 'rawvideo',
              '-pix_fmt', pixfmt,
              outputdir]
          err = utils.start_process_expect_returncode_0(cmd, wait=True)
          assert err == 0, "convert png to yuv failed."
      return output_original_frame_indices
      
  def _ext_file_yuv(self, input_fname, output_fname, item, pic_all_index, num_frames):
      height = item.args.SourceHeight
      width = item.args.SourceWidth
      bytes_per_pixel = 2 if item.args.InputBitDepth > 8 else 1

      # Determine sizes based on format
      if item.args.InputChromaFormat == "420":
          y_size = width * height
          uv_size = (width // 2) * (height // 2)
      elif item.args.InputChromaFormat == "422":
          y_size = width * height
          uv_size = (width // 2) * height
      elif item.args.InputChromaFormat == "444":
          y_size = width * height
          uv_size = width * height
      else:
          raise ValueError("Unsupported chroma format: {}".format(item.args.InputChromaFormat))

      vcmrs.log(f'_ext_file_yuv: input_fname:{input_fname}')
      vcmrs.log(f'_ext_file_yuv: output_fname:{output_fname}')
      vcmrs.log(f'_ext_file_yuv: y_size:{y_size}')
      vcmrs.log(f'_ext_file_yuv: uv_size:{uv_size}')
      vcmrs.log(f'_ext_file_yuv: item.args.InputBitDepth:{item.args.InputBitDepth}')
      vcmrs.log(f'_ext_file_yuv: bytes_per_pixel:{bytes_per_pixel}')

      input_file = open(input_fname, 'rb')
      output_file = open(output_fname, 'wb')

      output_original_frame_indices = []
      idx = 0
      ext_num_frames = 0
      while True:
          # read Y, U, V
          buffer = input_file.read((y_size + uv_size * 2) * bytes_per_pixel)
          if len(buffer) == 0: break
          if (idx >= item.args.FrameSkip + num_frames): break
          if (idx >= item.args.FrameSkip):
              if (idx - item.args.FrameSkip) in pic_all_index:
                  vcmrs.debug(f"need frame idx:{idx - item.args.FrameSkip}")
                  output_file.write(buffer)
                  ext_num_frames += 1
                  output_original_frame_indices.append( item.OriginalFrameIndices[idx] )
          idx += 1

      vcmrs.log(f'_ext_file_yuv: num_frames:{num_frames}')
      vcmrs.log(f'_ext_file_yuv: ext_num_frames:{ext_num_frames}')
      return output_original_frame_indices 

  def handle_temporal_segment_info(self,item):
      item.args.SegmentTemporalResampleMethod = []
      item.args.SegmentLength = []
      item.args.SegmentTemporalExtrapolationResampleNum = []
      item.args.SegmentTemporalExtrapolationPredictNum = []
      item.args.SegmentTemporalScale = [] 

      SegmentTemporalParams = []  # The number of elements is the same as the number of temporal segments
      pic_all_index = []
      param_dict={}
      for k in ['VCMEnabled', 'TemporalEnabled',
                'srdTemporalRestorationMode', 'srdTemporalInterpolationRatioId', 'srdTemporalExtraResamplingNumId', 'srdTemporalExtraPredictNumId','TemporalRemain',
                'PHTemporalChangedFlags',
                'PHTemporalRestorationMode','PHTemporalRatioIndexes', 'PHTemporalExtrapolationResampleNumIndexes','PHTemporalExtrapolationPredictNumIndexes',
                'TemporalResamplingPostHintFlag', 'trph_quality_valid_flag', 'trph_quality_value']:
          param_dict[k] = None
      TemporalEnabled = 1 if item.args.TemporalResample != 'Bypass' else 0
      param_dict["VCMEnabled"]=1
      param_dict['TemporalEnabled'] = TemporalEnabled

      # TemporalResamplingPostHintFlag condition 
      if item.args.TemporalResample == 'Bypass' or item.args.ROI == 'Bypass' or item.args.BitDepthTruncation == 'Bypass': # or item.args.SpatialResample == 'SimpleDownsample':
        item.args.TemporalResamplingPostHintFlag = 0 

      if len(item.args.SegmentLength) == 0 and len(item.args.SegmentTemporalResampleMethod) == 0:
          SegmentLength = [item.args.FramesToBeEncoded]
          if item.args.TemporalResampleMethod == 0:
              extRate = item.args.TemporalScale
              # for IntraPeriod downsample, down-sampled IntraPeriod should be compatible with GOP length
              # currently, ramdom access GOP configurations support not smaller than 8
              if item.IntraPeriod > 1 and int(item.IntraPeriod / item.args.TemporalScale) < 8:
                  extRate = int(item.IntraPeriod / 8)
              item.args.TemporalScale = extRate
              SegmentTemporalParams = [[0, extRate]]
              mixTemporalScale = extRate
              for i in range(item.args.FramesToBeEncoded):
                  if i % extRate ==0:
                      pic_all_index.append(i)
          else:
              SegmentTemporalParams = [
                  [1, item.args.TemporalExtrapolationResampleNum, item.args.TemporalExtrapolationPredictNum]]
              mixTemporalScale = (item.args.TemporalExtrapolationResampleNum+item.args.TemporalExtrapolationPredictNum)/item.args.TemporalExtrapolationResampleNum
              for i in range(item.args.FramesToBeEncoded):
                  tmp_list = [k for k in range(1, item.args.TemporalExtrapolationResampleNum + 1)]
                  if (i + 1) % (item.args.TemporalExtrapolationResampleNum+item.args.TemporalExtrapolationPredictNum) in tmp_list:
                      pic_all_index.append(i)
              # TemporalResamplingPostHintFlag only supports temporal resampling (interpolation) 
              item.args.TemporalResamplingPostHintFlag = 0 
          param_dict["PHTemporalChangedFlags"] = [0 for _ in range(len(pic_all_index))]
      else:
          SegmentLength = item.args.SegmentLength
          assert sum(
              item.args.SegmentLength) == item.args.FramesToBeEncoded, "The total length of the video sequence in item.args.SegmentLength should be the same as num_frames."
          assert len(item.args.SegmentLength) == len(item.args.SegmentTemporalResampleMethod), "The length of the list item.args.SegmentLength must be the same as the length of the list item.args.SegmentTemporalResampleMethod."
          SegmentTemporalScale = item.args.SegmentTemporalScale
          SegmentTemporalExtrapolationResampleNum = item.args.SegmentTemporalExtrapolationResampleNum
          SegmentTemporalExtrapolationPredictNum = item.args.SegmentTemporalExtrapolationPredictNum
          mixTemporalScale = 0
          PHTemporalChangedFlags = []
          PHTemporalRestorationMode = []
          PHTemporalRatioIndexes =[]
          PHTemporalExtrapolationResampleNumIndexes = []
          PHTemporalExtrapolationPredictNumIndexes = []
          # Currently TemporalResamplingPostHintFlag is disabled for PHTemporalChangedFlags 'on'  
          item.args.TemporalResamplingPostHintFlag = 0 
          for seg_idx in range(len(item.args.SegmentLength)):
              seg_temporal_method = item.args.SegmentTemporalResampleMethod[seg_idx]
              if seg_temporal_method==0:
                  # interpolation
                  extRate = SegmentTemporalScale.pop(0) if len(SegmentTemporalScale)>0 else item.args.TemporalScale
                  SegmentTemporalParams.append([0,extRate])
                  mixTemporalScale += extRate*(item.args.SegmentLength[seg_idx]/item.args.FramesToBeEncoded)
              else:
                  # extrapolation
                  TemporalExtrapolationResampleNum = SegmentTemporalExtrapolationResampleNum.pop(0) if len(SegmentTemporalExtrapolationResampleNum)>0 else item.args.TemporalExtrapolationResampleNum
                  TemporalExtrapolationPredictNum = SegmentTemporalExtrapolationPredictNum.pop(0) if len(SegmentTemporalExtrapolationPredictNum)>0 else item.args.TemporalExtrapolationPredictNum
                  SegmentTemporalParams.append([1, TemporalExtrapolationResampleNum,TemporalExtrapolationPredictNum])
                  mixTemporalScale += ((TemporalExtrapolationResampleNum+TemporalExtrapolationPredictNum)/TemporalExtrapolationResampleNum) * (item.args.SegmentLength[seg_idx]/item.args.FramesToBeEncoded)
                  # TemporalResamplingPostHintFlag only supports temporal resampling (interpolation)
                  item.args.TemporalResamplingPostHintFlag = 0 
              start_index = sum(item.args.SegmentLength[0:seg_idx]) if seg_idx!=0 else 0
              end_index = sum(item.args.SegmentLength[0:(seg_idx+1)])
              for i in range(start_index,end_index):
                  new_i = i - start_index
                  if seg_temporal_method == 0:
                      extRate = SegmentTemporalParams[-1][1]
                      if new_i % extRate == 0:
                          pic_all_index.append(i)
                          if seg_idx != 0 and new_i == 0:
                              PHTemporalChangedFlags.append(1)
                              PHTemporalRestorationMode.append(0)
                              PHTemporalRatioIndexes.append(resample_data.scale_factor_id_mapping[extRate])
                          else:
                              PHTemporalChangedFlags.append(0)
                  else:
                      TemporalExtrapolationResampleNum, TemporalExtrapolationPredictNum = SegmentTemporalParams[-1][1],SegmentTemporalParams[-1][2]
                      tmp_list = [k for k in range(1, TemporalExtrapolationResampleNum + 1)]
                      if (new_i + 1) % (
                              TemporalExtrapolationResampleNum + TemporalExtrapolationPredictNum) in tmp_list:
                          pic_all_index.append(i)
                          if seg_idx != 0 and new_i == 0:
                              PHTemporalChangedFlags.append(1)
                              PHTemporalRestorationMode.append(1)
                              PHTemporalExtrapolationResampleNumIndexes.append(resample_data.resample_len_id_mapping[TemporalExtrapolationResampleNum])
                              PHTemporalExtrapolationPredictNumIndexes.append(resample_data.predict_len_id_mapping[TemporalExtrapolationPredictNum])
                          else:
                              PHTemporalChangedFlags.append(0)
          param_dict["PHTemporalChangedFlags"]=PHTemporalChangedFlags
          param_dict["PHTemporalRestorationMode"]=PHTemporalRestorationMode
          param_dict["PHTemporalRatioIndexes"]=PHTemporalRatioIndexes
          param_dict["PHTemporalExtrapolationResampleNumIndexes"]=PHTemporalExtrapolationResampleNumIndexes
          param_dict["PHTemporalExtrapolationPredictNumIndexes"]=PHTemporalExtrapolationPredictNumIndexes
      first_seg_param = SegmentTemporalParams[0]
      param_dict["srdTemporalRestorationMode"]= first_seg_param[0]
      if first_seg_param[0]==0:
          param_dict["srdTemporalInterpolationRatioId"] = resample_data.scale_factor_id_mapping[first_seg_param[1]]
      else:
          param_dict["srdTemporalExtraResamplingNumId"] = resample_data.resample_len_id_mapping[first_seg_param[1]]
          param_dict["srdTemporalExtraPredictNumId"] = resample_data.predict_len_id_mapping[first_seg_param[2]]
      param_dict["TemporalRemain"] = item.args.FramesToBeEncoded - pic_all_index[-1]-1
      param_dict["TemporalResamplingPostHintFlag"] = item.args.TemporalResamplingPostHintFlag

      return SegmentLength,mixTemporalScale,pic_all_index,param_dict

  def process(self, input_fname, output_fname, item, ctx):
    video_info = video_utils.get_video_info(input_fname, item.args)
    num_frames = video_info.num_frames - item.args.FrameSkip
    if item.args.FramesToBeEncoded > 0: num_frames = min(num_frames, item.args.FramesToBeEncoded)
    item.args.FramesToBeEncoded = num_frames

    SegmentLength, mixTemporalScale, pic_all_index, param_dict = self.handle_temporal_segment_info(item)
    if item._is_dir_video:
        item.OriginalFrameIndices = self._ext_files(input_fname, output_fname, pic_all_index, item, video_info)
        self.write_cfg(item, param_dict)
        self._updateExtArgs(item, len(pic_all_index), mixTemporalScale)
        # self._set_parameter(item, extRate, num_frames)
    elif item._is_yuv_video:
        # default the output format of temporal resample is the same as the input
        perform_extraction_in_yuv = True
        if perform_extraction_in_yuv:
          item.OriginalFrameIndices = self._ext_file_yuv(input_fname, output_fname, item,pic_all_index, num_frames)
        else:
          timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
          pngpath = os.path.join(os.path.abspath(item.args.working_dir), os.path.basename(input_fname), 'tmp', f'{item.args.quality}_{timestamp}')
          makedirs(pngpath)
          yuvTopngs(input_fname, item, pngpath, num_frames, video_info)
          item.OriginalFrameIndices = self._ext_files(pngpath, output_fname, pic_all_index, item, video_info, yuvFile=True)
        self.write_cfg(item, param_dict)
        self._updateExtArgs(item, len(pic_all_index), mixTemporalScale)
    elif os.path.isfile(input_fname) and (os.path.splitext(input_fname)[1].lower() in ['.png', '.jpg', '.jpeg']):
        rm_file_dir(output_fname)
        enforce_symlink(input_fname, output_fname)
        self.write_cfg(item, param_dict)
    else:
      assert False, f"Input file {input_fname} is not found"


  def write_cfg(self, item, param_dict):
    for k,v in param_dict.items():
        vcmrs.debug(f"{k}:{v}")
    video_working_dir = item.working_dir
    makedirs(video_working_dir)
    outputfile = item.trd_enc_fname
    
    if os.path.exists(outputfile):
       os.remove(outputfile)

    f = open(outputfile, 'w+')
    content = "#======== extentsion =====================\n"
    content +=f"VCMEnabled:{param_dict['VCMEnabled']}\n"
    content +=f"TemporalEnabled:{param_dict['TemporalEnabled']}\n"
    content +=f"srdTemporalRestorationMode:{param_dict['srdTemporalRestorationMode']}\n"
    if param_dict["srdTemporalRestorationMode"]==0:
        content += f"srdTemporalInterpolationRatioId:{param_dict['srdTemporalInterpolationRatioId']}\n"
    elif param_dict["srdTemporalRestorationMode"]==1:
        content += f"srdTemporalExtraResamplingNumId:{param_dict['srdTemporalExtraResamplingNumId']}\n"
        content += f"srdTemporalExtraPredictNumId:{param_dict['srdTemporalExtraPredictNumId']}\n"
    content += f"TemporalRemain:{param_dict['TemporalRemain']}\n"
    content += f"PHTemporalChangedFlags:{','.join(str(flag) for flag in param_dict['PHTemporalChangedFlags'])}\n"

    if param_dict["PHTemporalRestorationMode"] is not None and len(param_dict["PHTemporalRestorationMode"]) > 0:
        content += f"PHTemporalRestorationMode:{','.join(str(flag) for flag in param_dict['PHTemporalRestorationMode'])}\n"
    if param_dict["PHTemporalRatioIndexes"] is not None and len(param_dict["PHTemporalRatioIndexes"]) > 0:
        content += f"PHTemporalRatioIndexes:{','.join(str(flag) for flag in param_dict['PHTemporalRatioIndexes'])}\n"
    if param_dict["PHTemporalExtrapolationResampleNumIndexes"] is not None and len(param_dict["PHTemporalExtrapolationResampleNumIndexes"]) > 0:
        content += f"PHTemporalExtrapolationResampleNumIndexes:{','.join(str(flag) for flag in param_dict['PHTemporalExtrapolationResampleNumIndexes'])}\n"
    if param_dict["PHTemporalExtrapolationPredictNumIndexes"] is not None and len(param_dict["PHTemporalExtrapolationPredictNumIndexes"]) > 0:
        content += f"PHTemporalExtrapolationPredictNumIndexes:{','.join(str(flag) for flag in param_dict['PHTemporalExtrapolationPredictNumIndexes'])}\n"
    content +=f"TemporalResamplingPostHintFlag:{param_dict['TemporalResamplingPostHintFlag']}\n" 
    f.write(content)
    f.close()

  def _set_parameter(self, item, extRate, FramesToBeEncoded):
    # sequence level parameter
    #default scale: 1 byte, framestoberestore: 2 bytes
    param_data = bytearray(3)
    param_data[0] = self.scale_factor_mapping[extRate]
    param_data[1] = (FramesToBeEncoded >> 8) & 0xFF
    param_data[2] = FramesToBeEncoded & 0xFF
    item.add_parameter('TemporalResample', param_data=param_data)
    vcmrs.log(f'encoder extRate:{extRate}, FramesToBeRecon:{FramesToBeEncoded}')
    pass
