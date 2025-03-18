# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import cv2
import datetime
import time
import numpy as np
import shutil
import vcmrs
from vcmrs.Utils.io_utils import enforce_symlink, rm_file_dir, makedirs
from vcmrs.Utils.component import Component
from . import resample_data
from vcmrs.Utils import utils
from . import Extrapolation
from . import Interpolation

class resample(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    self.id_scale_factor_mapping = resample_data.id_scale_factor_mapping

  def _updateExtArgs(self, item, num_frames,qps_list,mixRate):
    item.FrameRateRelativeVsInnerCodec = item.FrameRateRelativeVsInnerCodec * mixRate

    item.video_info.num_frames = num_frames

    # AllIntra Mode
    if len(item.video_info.intra_indices)>1 and (item.video_info.intra_indices[1] - item.video_info.intra_indices[0])==1:
      item.video_info.intra_indices = list(range(num_frames))
    else:
      for k in range(len(item.video_info.intra_indices)):
        item.video_info.intra_indices[k] = int(item.video_info.intra_indices[k] * mixRate)
    item.video_info.frame_qps = qps_list

  def write_cfg(self, item):
    infile = item.trd_enc_fname
    outputfile = item.trd_dec_fname
    if os.path.exists(infile):
      with open(infile, 'r+') as f:
        lines = f.readlines()

      lines = [i.strip() for i in lines if i.strip() != "" and i.startswith('#') == False]
      outf = open(outputfile, 'w+')
      outf.write('\n'.join(lines))
      outf.close()

  def read_cfg(self, cfg_path):
    new_param_dict = {}
    if os.path.exists(cfg_path):
      with open(cfg_path, 'r+') as f:
        lines = f.readlines()

      lines = [i.strip() for i in lines if i.strip() != ""]
      param_dict = {}
      for i in range(len(lines)):
        if ":" in lines[i]:
          key, value = lines[i].split(':')
          key = key.strip()
          value = value.strip()
          value_list = value.split(",")
          if len(value_list)>1 or key in ["PHTemporalChangedFlags","PHTemporalRestorationMode","PHTemporalRatioIndexes","PHTemporalExtrapolationResampleNumIndexes","PHTemporalExtrapolationPredictNumIndexes", "trph_quality_valid_flag", "trph_quality_value"]:
            param_dict[key.strip()] = [int(z) for z in value_list]
          else:
            param_dict[key.strip()] = int(value.strip())

      if "TemporalEnabled" in param_dict.keys():
        new_param_dict["TemporalEnabled"] = param_dict["TemporalEnabled"]
      if "srdTemporalRestorationMode" in param_dict.keys():
        new_param_dict["srdTemporalRestorationMode"] = param_dict["srdTemporalRestorationMode"]
        if param_dict["srdTemporalRestorationMode"]==0:
          if "srdTemporalInterpolationRatioId" in param_dict.keys():
            new_param_dict["srdTemporalResamplingRatio"] = resample_data.id_scale_factor_mapping[param_dict["srdTemporalInterpolationRatioId"]]
        elif param_dict["srdTemporalRestorationMode"]==1:
          if "srdTemporalExtraResamplingNumId" in param_dict.keys():
            new_param_dict["srdTemporalExtrapolationResampleNum"] = resample_data.id_resample_len_mapping[
              param_dict["srdTemporalExtraResamplingNumId"]]
          if "srdTemporalExtraPredictNumId" in param_dict.keys():
            new_param_dict["srdTemporalExtrapolationPredictNum"] = resample_data.id_predict_len_mapping[
              param_dict["srdTemporalExtraPredictNumId"]]
      if "TemporalRemain" in param_dict.keys():
        new_param_dict["TemporalRemain"] = param_dict["TemporalRemain"]
      if "PHTemporalChangedFlags" in param_dict.keys():
        new_param_dict["PHTemporalChangedFlags"] = param_dict["PHTemporalChangedFlags"]
      if "PHTemporalRestorationMode" in param_dict.keys():
        new_param_dict["PHTemporalRestorationMode"] = param_dict["PHTemporalRestorationMode"]
      if "PHTemporalRatioIndexes" in param_dict.keys():
        new_param_dict["PHTemporalRatios"] = [resample_data.id_scale_factor_mapping[i] for i in param_dict["PHTemporalRatioIndexes"]]
      if "PHTemporalExtrapolationResampleNumIndexes" in param_dict.keys():
        new_param_dict["PHTemporalExtrapolationResampleNums"] = [resample_data.id_resample_len_mapping[i] for i in param_dict["PHTemporalExtrapolationResampleNumIndexes"]]
      if "PHTemporalExtrapolationPredictNumIndexes" in param_dict.keys():
        new_param_dict["PHTemporalExtrapolationPredictNums"] = [resample_data.id_predict_len_mapping[i] for i in param_dict["PHTemporalExtrapolationPredictNumIndexes"]]
      if "TemporalResamplingPostHintFlag" in param_dict.keys():
        new_param_dict["TemporalResamplingPostHintFlag"] = param_dict["TemporalResamplingPostHintFlag"]
      if "trph_quality_valid_flag" in param_dict.keys():
        new_param_dict["trph_quality_valid_flag"] = [i for i in param_dict["trph_quality_valid_flag"]]        
      if "trph_quality_value" in param_dict.keys():
        new_param_dict["trph_quality_value"] = [i for i in param_dict["trph_quality_value"]]             
      for k,v in new_param_dict.items():
        vcmrs.debug(f"{k}:{v}")
    return new_param_dict

  def get_new_qp(self, frame_qps_list, i):
    i = i-1
    if frame_qps_list[i] < 0 and frame_qps_list[i + 1] < 0:
      newQP = frame_qps_list[i]
    elif min(frame_qps_list[i],frame_qps_list[i+1])<0:
      newQP = max(frame_qps_list[i], frame_qps_list[i+1])
    else:
      newQP = round((frame_qps_list[i]+frame_qps_list[i + 1])/2)
    return newQP

  def handle_picture_level(self, param_dict,frame_qps_list,bFilter=False,pngpath=None, output_fname_png=None,yuv_input=None,yuv_output=None,if_yuv=True):
    if if_yuv:
      assert yuv_input is not None and yuv_output is not None, "When if_yuv is true, yuv_input and yuv_output must be passed"
    else:
      assert pngpath is not None and output_fname_png is not None, "When if_yuv is false, pngpath and output_fname_png must be passed"

    # create output folder
    if output_fname_png is not None:
      if os.path.exists(output_fname_png):
        if os.path.isdir(output_fname_png):
          shutil.rmtree(output_fname_png)
        else:
          os.remove(output_fname_png)
      makedirs(output_fname_png)

    if not if_yuv:
      if not os.path.isdir(pngpath):
        enforce_symlink(pngpath, output_fname_png)
        return 1, frame_qps_list,1

    videogen = []
    if not if_yuv:
      for f in sorted(os.listdir(pngpath)):
          if "png" in f:
              videogen.append(os.path.join(pngpath, f))
      if bFilter:
        # do the filter before temporal up-resample
        srcfile = videogen[0]
        img = cv2.imread(srcfile)
        kernel = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
        img3 = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(srcfile, img3)
    PHTemporalChangedFlags = param_dict["PHTemporalChangedFlags"] if "PHTemporalChangedFlags" in param_dict.keys() else []
    PHTemporalRestorationMode = param_dict["PHTemporalRestorationMode"] if "PHTemporalRestorationMode" in param_dict.keys() else []
    PHTemporalRatios = param_dict["PHTemporalRatios"] if "PHTemporalRatios" in param_dict.keys() else []
    PHTemporalExtrapolationResampleNums = param_dict["PHTemporalExtrapolationResampleNums"] if "PHTemporalExtrapolationResampleNums" in param_dict.keys() else []
    PHTemporalExtrapolationPredictNums = param_dict["PHTemporalExtrapolationPredictNums"] if "PHTemporalExtrapolationPredictNums" in param_dict.keys() else []
    TemporalRemain = param_dict["TemporalRemain"]
    temporal_mode = param_dict["srdTemporalRestorationMode"]
    if temporal_mode==0:
      temporal_param = [param_dict["srdTemporalResamplingRatio"]]
      mixRate = param_dict["srdTemporalResamplingRatio"]
    else:
      temporal_param = [param_dict["srdTemporalExtrapolationResampleNum"],param_dict["srdTemporalExtrapolationPredictNum"]]
      mixRate = (param_dict["srdTemporalExtrapolationResampleNum"]+param_dict["srdTemporalExtrapolationPredictNum"])/param_dict["srdTemporalExtrapolationResampleNum"]
    frameBuffer = None
    yuvBuffer = None
    index = 0
    extrapolation_model = None
    interpolation_model = None
    stack_len_for_extrapolation = 1
    qps_list = []
    index_for_sampled = 0
    while True:
      try:
        if temporal_mode==0:
          if if_yuv:
            ret, frame, yuv = yuv_input.read()
            if not ret:
              raise IndexError
          else:
            frame_path = videogen.pop(0)
            frame = cv2.imread(frame_path)[:, :, ::-1].copy()
          # interpolation
          if frameBuffer is not None:
            interpolation_ratio=temporal_param[0]
            if interpolation_model is None:
              interpolation_model = Interpolation.load_model()
            Interpolation.interpolation_frame_by_frame(interpolation_model,frameBuffer,frame,interpolation_ratio,(None, None),output_fname_png,index,outputyuv=yuv_output)
            assert index_for_sampled!=0, "index_for_sampled!=0"
            new_qp = self.get_new_qp(frame_qps_list,index_for_sampled)
            qps_list.extend([new_qp]*(interpolation_ratio-1))
            index = index + interpolation_ratio-1
          frameBuffer = frame
          if if_yuv:
            yuvBuffer = yuv
            yuv_output.write_single_frame(yuv)
          else:
            target_path = os.path.join(output_fname_png, 'frame_{:0>6d}.png'.format(index))
            os.symlink(frame_path, target_path)
          qps_list.append(frame_qps_list[index_for_sampled])
          index = index + 1

          TemporalChangedFlag = PHTemporalChangedFlags.pop(0) if len(PHTemporalRatios)>0 else 0
          if TemporalChangedFlag == 1:
            temporal_mode = PHTemporalRestorationMode.pop(0)
            if temporal_mode == 0:
              temporal_param = [PHTemporalRatios.pop(0)]
            else:
              temporal_param = [PHTemporalExtrapolationResampleNums.pop(0), PHTemporalExtrapolationPredictNums.pop(0)]
              stack_len_for_extrapolation +=1
        else:
          if if_yuv:
            ret, frame, yuv = yuv_input.read(reverse=False)
            if not ret:
              raise IndexError
          else:
            frame_path = videogen.pop(0)
            frame = cv2.imread(frame_path)
          if if_yuv:
            yuv_output.write_single_frame(yuv)
          else:
            target_path = os.path.join(output_fname_png, 'frame_{:0>6d}.png'.format(index))
            os.symlink(frame_path, target_path)
          qps_list.append(frame_qps_list[index_for_sampled])
          index = index + 1

          TemporalChangedFlag = PHTemporalChangedFlags.pop(0) if len(PHTemporalRatios)>0 else 0
          if TemporalChangedFlag == 1:
            temporal_mode = PHTemporalRestorationMode.pop(0)
            if temporal_mode == 0:
              temporal_param = [PHTemporalRatios.pop(0)]
            else:
              temporal_param = [PHTemporalExtrapolationResampleNums.pop(0), PHTemporalExtrapolationPredictNums.pop(0)]
          if temporal_mode==0:
            # interpolation
            assert frameBuffer is None, "At this moment, frameBuffer must be None"
            frameBuffer = frame
            if if_yuv:
              yuvBuffer = yuv
            continue
          else:
            # extrapolation
            if stack_len_for_extrapolation == temporal_param[0] and frameBuffer is not None:
              need_frame =  temporal_param[1]
              if if_yuv:
                if yuv_input.ori_frame_num==index_for_sampled+1:
                  need_frame = TemporalRemain
              else:
                if len(videogen)==0:
                  need_frame = TemporalRemain
              if extrapolation_model is None:
                extrapolation_model = Extrapolation.load_model()
              Extrapolation.extrapolation_frames_by_frames(extrapolation_model, frameBuffer, frame, need_frame,
                                                           output_fname_png, index,outputyuv=yuv_output)
              qps_list.extend([frame_qps_list[index_for_sampled]]*need_frame)
              index += need_frame
              stack_len_for_extrapolation = 1
              frameBuffer=None
            else:
              if stack_len_for_extrapolation == (temporal_param[0]-1):
                frameBuffer = frame
                if if_yuv:
                  yuvBuffer = yuv
              stack_len_for_extrapolation += 1
      except IndexError:
        break
      index_for_sampled +=1
    if temporal_mode==0:
      if if_yuv:
        for _ in range(TemporalRemain):
          yuv_output.write_single_frame(yuvBuffer)
          index = index + 1
          qps_list.append(qps_list[-1])
      else:
        last_sampled_frame = os.path.join(output_fname_png, 'frame_{:0>6d}.png'.format(index-1))
        for _ in range(TemporalRemain):
          target_path = os.path.join(output_fname_png, 'frame_{:0>6d}.png'.format(index))
          os.symlink(last_sampled_frame, target_path)
          index = index + 1
          qps_list.append(qps_list[-1])
    if "PHTemporalChangedFlags" in param_dict.keys() and 1 in param_dict["PHTemporalChangedFlags"]:
      mixRate = index/index_for_sampled
    return index,qps_list,mixRate
  
  def get_TemporalResamplingPostHintParameters(self, item, param_dict): 
    item.args.TemporalResamplingPostHintFlag = param_dict["TemporalResamplingPostHintFlag"] 
    if item.args.TemporalResamplingPostHintFlag: 
      score_path = os.path.dirname(item.working_dir)
      if ("trph_quality_valid_flag" in param_dict) and ("trph_quality_value" in param_dict): 
        score_flag_list = [i for i in param_dict["trph_quality_valid_flag"]]  
        score_list = [i for i in param_dict["trph_quality_value"]]  
        if len(score_flag_list) != 0 and len(score_list) != 0: 
          makedirs(f"{score_path}/score_enc")
          flag_file = open(f"{score_path}/score_enc/score_flag_list.txt", 'w')
          score_file = open(f"{score_path}/score_enc/score_list.txt", 'w')
          for flag, score in zip(score_flag_list, score_list):
              flag_file.write(f"{flag}\n")
              score_file.write(f"{score}\n")
          flag_file.close()
          score_file.close()

  def process(self, input_fname, output_fname, item, ctx):
    vcmrs.log('######decode temporal process###########')
    starttime = time.time()
    # default_scale_factor, FramesToBeRecon = self._get_parameter(item)

    if item._is_yuv_video or item._is_dir_video:
      if hasattr(item, 'trd_enc_fname'):
        self.write_cfg(item)
      # read cfg
    if hasattr(item, 'trd_dec_fname'):
      param_dict  = self.read_cfg(item.trd_dec_fname)
    else:
      param_dict = {}

    if len(param_dict) == 0 or ((len(param_dict) != 0)and( "TemporalEnabled" not in param_dict.keys())) or ((len(param_dict) != 0)and(param_dict["TemporalEnabled"]==0)):
      # scale_factor and FramesToBeRecon will be None when the encoder doesn't set temporal parameters
      # link the output to input directly
      makedirs(os.path.dirname(output_fname))
      rm_file_dir(output_fname)
      enforce_symlink(input_fname, output_fname)
      return

    if item._is_dir_video:
      # input format ： directory
      pngpath = input_fname
      output_fname_png = output_fname
      bFilter = True
      if item.args.InnerCodec == 'VTM':
        bFilter = False
      FramesToBeRecon, qps_list, mixRate = self.handle_picture_level(param_dict, item.video_info.frame_qps,bFilter=bFilter,
                                                                     pngpath=pngpath, output_fname_png=output_fname_png,yuv_input=None,yuv_output=None,if_yuv=False)
      self._updateExtArgs(item, FramesToBeRecon, qps_list, mixRate)
      # self.get_TemporalResamplingPostHintParameters(item, param_dict) # current only yuv 
    elif item._is_yuv_video:
      H, W, C = item.video_info.resolution

      yuv_input = Interpolation.VideoCaptureYUV(input_fname, (H, W), mode='read')
      makedirs(os.path.dirname(output_fname))
      yuv_output = Interpolation.VideoCaptureYUV(output_fname, (H, W), mode='write')
      FramesToBeRecon, qps_list, mixRate = self.handle_picture_level(param_dict,item.video_info.frame_qps,
                                                                     pngpath=None, output_fname_png=None,yuv_input=yuv_input,yuv_output=yuv_output,if_yuv=True)
      vcmrs.log(f"FramesToBeRecon: {FramesToBeRecon}")
      self._updateExtArgs(item, FramesToBeRecon, qps_list, mixRate)
      self.get_TemporalResamplingPostHintParameters(item, param_dict)
    else:
      assert False, f"Input file {input_fname} is not found"
    durationtime = time.time() - starttime
    vcmrs.log(f'Temporal decoding completed in {durationtime} seconds')

  def _resize_frame(self, input_fname, output_fname, scale_factor, FramesToBeRecon, bFilter):
    #default input_fname and output_fname are both directory
    rm_file_dir(output_fname)
    if scale_factor == 1:
      rm_file_dir(output_fname)
      makedirs(output_fname)
      for file in os.listdir(input_fname):
        idx = int(os.path.basename(file).split('.')[0].split('_')[-1])
        srcfile = os.path.join(input_fname, file)
        desfile = os.path.join(output_fname, 'frame_%06d.png'%(idx)) # %07d.png
        rm_file_dir(desfile)
        enforce_symlink(srcfile, desfile) 
    else:
      if bFilter:
        # do the filter before temporal up-resample
        first = sorted(os.listdir(input_fname))[0]
        srcfile = os.path.join(input_fname, first)
        img = cv2.imread(srcfile)
        kernel = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
        img3 = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(srcfile, img3)
        # do the temporal up-resample
      Interpolation(input_fname, output_fname, scale_factor, FramesToBeRecon)

  def _get_parameter(self, item):
    # sequence level parameter
    scale_factor = None
    FramesToBeRecon = None
    param_data = item.get_parameter('TemporalResample')
    if param_data is not None:
      assert len(param_data) == 3, f'received parameter data is not correct: {param_data}'
      scale_factor = self.id_scale_factor_mapping[param_data[0]]
      FramesToBeRecon = (param_data[1] << 8) + param_data[2]
      vcmrs.log(f'decoder scale_factor:{scale_factor}, FramesToBeRecon:{FramesToBeRecon}')
    return scale_factor, FramesToBeRecon
