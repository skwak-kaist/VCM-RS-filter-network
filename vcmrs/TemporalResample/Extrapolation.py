### based on CVPR2023_DMVFN use I0 and I1 to predict I2
import os
import cv2
import torch
import warnings
import vcmrs
from pathlib import Path
import numpy as np
from torch.nn import functional as F
from .models_extrapolation.model import Model
import vcmrs.TemporalResample.models_extrapolation as temporal_model
temporal_model_dir = Path(temporal_model.__path__[0])
torch.use_deterministic_algorithms(True)

def saveImg(item,savepath,index):
    tmpname = os.path.join(savepath, 'frame_{:0>6d}.png'.format(index))
    cv2.imwrite(tmpname, item)

def load_model():
    ###
    warnings.filterwarnings("ignore")
    # load model
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    model = Model(load_path=os.path.join(temporal_model_dir, "efnet.pkl"))
    model.eval()
    vcmrs.log('Temporal extrapolation model loaded.')
    return model

def preprocess(frame1, frame2):
    device = torch.device("cpu")
    if frame1 is not None:
        h, w, _ = frame1.shape
    else:
        h, w, _ = frame2.shape
    tmp = max(32, int(32 / 1.0))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    if frame1 is not None:
        I1 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 256.
        frame1 = F.pad(I1, padding)
    if frame2 is not None:
        I2 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 256.
        frame2 = F.pad(I2, padding)
    return frame1, frame2,h,w,padding

def extrapolation_frames_by_frames(model, frameBuffer, frame, need_frame, savepath, index,outputyuv=None):
    frame1, frame2,h,w,padding = preprocess(frameBuffer, frame)
    I0 = frame1
    I1 = frame2
    for i in range(need_frame):
        predicted_frame_ori = model.predict_frame(I0, I1, padding)
        predicted_frame = (((predicted_frame_ori.squeeze() * 256.).byte().cpu().numpy().transpose(1, 2, 0)))
        if outputyuv is not None:
            outputyuv.write_single_frame_rgb(predicted_frame[:h, :w], (h, w),reverse=False)
        else:
            saveImg(predicted_frame[:h, :w],savepath,index)
        index += 1
        I0 = I1
        I1 = predicted_frame_ori[:h, :w]