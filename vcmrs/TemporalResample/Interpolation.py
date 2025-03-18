# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import math,os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue
import shutil
import vcmrs
from .models.pytorch_msssim import ssim_matlab
from .models.CodeIF import Model
from pathlib import Path
import vcmrs.TemporalResample.models as temporal_model
from vcmrs.Utils.io_utils import makedirs
temporal_model_dir = Path(temporal_model.__path__[0])
torch.use_deterministic_algorithms(True)

import hashlib

def cal_md5(output_tensor, type='tensor'):
    if type == "tensor":
        output_numpy = output_tensor.detach().cpu().numpy()
    else:
        output_numpy = output_tensor
    md5 = hashlib.md5(output_numpy.data.tobytes(order='F'))
    return md5.hexdigest()

class VideoCaptureYUV:
    def __init__(self, filename, size, yuvformat="420", inputdepth=10, mode='read'):
        self.shape = size
        self.height, self.width = size

        if yuvformat == "420":
            y_size = self.height * self.width
            uv_size = (self.height // 2) * (self.width // 2)
            uv_shape = (self.height // 2, self.width // 2)
        elif yuvformat == "422":
            y_size = self.height * self.width
            uv_size = self.height * (self.width // 2)
            uv_shape = (self.height, self.width // 2)
        elif yuvformat == "444":
            y_size = self.height * self.width
            uv_size = self.height * self.width
            uv_shape = (self.height, self.width)
        else:
            raise ValueError("Please recheck yuv format.")

        self.bytes = 2 if inputdepth > 8 else 1
        self.dtype = np.uint16 if inputdepth > 8 else np.uint8
        self.frame_len = int((y_size + uv_size * 2) * self.bytes)
        self.y_size = y_size
        self.uv_size = uv_size
        self.uv_shape = uv_shape
        if mode == "read":
            self.f = open(filename, 'rb')
            self.ori_frame_num = self.cal_frame_num(filename)
        else:
            self.f = open(filename, 'wb')
            self.ori_frame_num = 0

    def cal_frame_num(self,filename):
        file_size = os.path.getsize(filename)
        if file_size % self.frame_len != 0:
            raise ValueError("The file size is not a multiple of the frame size.")
        return file_size//self.frame_len

    def upsample(self, arr, target_shape):
        outarr = np.zeros(target_shape)

        h, w = target_shape
        for i in range(h):
            for j in range(w):
                outarr[i, j] = arr[i // 2, j // 2]
        return outarr

    def _seek(self):
        self.f.seek(-(self.y_size + self.uv_size + self.uv_size) * self.bytes, 1)

    def read_raw(self):
        try:
            yuv_raw = self.f.read((self.y_size + self.uv_size + self.uv_size) * self.bytes)
            y = np.frombuffer(yuv_raw[:(self.y_size * self.bytes)], dtype=self.dtype).reshape((self.height, self.width))
            u = np.frombuffer(
                yuv_raw[(self.y_size * self.bytes):(self.uv_size * self.bytes + self.y_size * self.bytes)],
                dtype=self.dtype).reshape(self.uv_shape)
            v = np.frombuffer(yuv_raw[(self.uv_size * self.bytes + self.y_size * self.bytes):],
                              dtype=self.dtype).reshape(self.uv_shape)
            yuv_trans = np.frombuffer(yuv_raw, dtype=self.dtype).reshape(int(self.height * 3 / 2), self.width)
        except Exception as e:
            return False, None, None

        return True, (y, u, v), yuv_trans

    def read(self,reverse=True):
        ret, yuv, yuv_trans = self.read_raw()
        if not ret:
            return ret, None, None
        if reverse:
            rgb = cv2.cvtColor((yuv_trans / 1024 * 256).astype(np.uint8), cv2.COLOR_YUV2BGR_I420)[:, :, ::-1].copy()
        else:
            rgb = cv2.cvtColor((yuv_trans / 1024 * 256).astype(np.uint8), cv2.COLOR_YUV2BGR_I420).copy()
        return ret, rgb, yuv

    def write_single_frame(self, yuv):  # write frame in yuv format
        yuv_frame = [mat.tobytes() for mat in yuv]
        self.f.write(b''.join(yuv_frame))

    def write_single_frame_rgb(self, rgb, size, outbytes=10,reverse=True):  # write frame in rgb format
        h, w = size
        if reverse:
            rgb = rgb[:, :, ::-1]
        yuv_trans = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV_I420)  # COLOR_BGR2YUV_I420,
        y, u, v = yuv_trans[:h, :w], yuv_trans[h:(h + int(0.25 * h)), :], yuv_trans[(h + int(0.25 * h)):,
                                                                          :]  # I420 format
        u = u.reshape(-1, int(w / 2))
        v = v.reshape(-1, int(w / 2))

        if outbytes == 10:
            y = (y.astype(np.uint16) << 2)
            u = (u.astype(np.uint16) << 2)
            v = (v.astype(np.uint16) << 2)

        self.write_single_frame((y, u, v))  # class


def Interpolation_frame_by_frame(upresampler, frameBuffer, frame, scale_factor, temp, tail=0):
    output = []

    I0, I1 = upresampler.preprocess(frameBuffer, frame)
    temp_I0, temp_I1 = temp
    if temp_I0 is not None:
        I0 = temp_I0
    if temp_I1 is not None:
        I1 = temp_I1


    # not tail 
    if tail == 0 and frame is not None:
        middle = upresampler.make_inference(I0, I1, scale_factor - 1)   
        output += middle

    # tail and copy
    if tail != 0:   # tail currently is defined as the last interval including the last frame 
        for _ in range(tail):
            if I1 is not None:
                output.append(I1)
            else:
                output.append(I0)

    h, w, _ = frameBuffer.shape


    for mid in output:
        mid = (((mid[0] * 256.).byte().cpu().numpy().transpose(1, 2, 0)))
        upresampler.saveImg(mid[:h, :w])


def Interpolation(imgspath, interpolateDir, rate, FramesToBeRecon):
    savepath = interpolateDir
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    makedirs(savepath)

    exp = math.log2(rate)
    cnt = (FramesToBeRecon-1) // rate
    upresampler = _UpResampler_()
    upresampler.setAttr(imgspath, int(exp), savepath, 0, cnt*rate)
    upresampler.infer()
    # padding end part with the last frame if necessary
    MOD = (FramesToBeRecon-1) % rate
    if MOD != 0:
        vcmrs.log(f'mod ={MOD}')
        lastName = sorted(os.listdir(savepath))[-1]
        last = lastName.split('.')[0].split('_')[-1]
        cnt = len(last)
        for i in range(MOD):
            padName = f'%0{len(last)}d.png' % (int(last) + i + 1)
            cnt1 = len(lastName.split('.')[0]) - cnt
            padName = lastName[0:cnt1] + padName
            curpath = os.path.join(savepath, lastName)
            copyDes = os.path.join(savepath, padName)
            shutil.copy(curpath, copyDes)

def load_model():
    warnings.filterwarnings("ignore")
    # load model
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        # cancel dynamic algorithm
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    model = Model()
    model.load_model(temporal_model_dir, -1)
    model.eval()
    model.device()
    vcmrs.log('Temporal interpolation model loaded.')
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
    return frame1, frame2,padding,h,w

def make_inference(model, I0, I1, n):
    middle = model.inference(I0, I1, 1.0)
    if n == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, n=n//2)
    second_half = make_inference(model, middle, I1, n=n//2)

    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def saveImg(item,savepath,index):
    tmpname = os.path.join(savepath, 'frame_{:0>6d}.png'.format(index))
    cv2.imwrite(tmpname, item[:, :, ::-1])
def interpolation_frame_by_frame(model, frameBuffer, frame, scale_factor, temp, savepath, index, outputyuv=None):
    output = []

    I0, I1,padding,h,w = preprocess(frameBuffer, frame)
    temp_I0, temp_I1 = temp
    if temp_I0 is not None:
        I0 = temp_I0
    if temp_I1 is not None:
        I1 = temp_I1

    middle = make_inference(model, I0, I1, scale_factor - 1)
    output += middle

    for mid in output:
        mid = (((mid[0] * 256.).byte().cpu().numpy().transpose(1, 2, 0)))
        if outputyuv is not None:
            outputyuv.write_single_frame_rgb(mid[:h, :w], (h, w))
        else:
            saveImg(mid[:h, :w],savepath,index)
        index = index+1

class _UpResampler_:
    def __init__(self):
        warnings.filterwarnings("ignore")
        # load model
        self.device = torch.device("cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            #cancel dynamic algorithm
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        self.model = Model()
        self.model.load_model(temporal_model_dir, -1)
        self.model.eval()
        self.model.device()
        vcmrs.log('Temporal model loaded.')
    
    def setAttr(self, img, exp, savepath, startIdx, endIdx):
        self.cnt = 0
        self.img = img
        self.exp = int(exp)
        self.savepath = savepath
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.maxLen = int((self.endIdx - self.startIdx) // (2**self.exp)) + 1

    def saveImg(self, item):
        tmpname = os.path.join(self.savepath, 'frame_{:0>6d}.png'.format(self.cnt))
        cv2.imwrite(tmpname, item[:, :, ::-1])
        self.cnt += 1

    def preprocess(self, frame1, frame2):
        if frame1 is not None:
            h, w, _ = frame1.shape
        else:
            h, w, _ = frame2.shape
        tmp = max(32, int(32 / 1.0))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        if frame1 is not None:
            I1 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 256.
            frame1 = F.pad(I1, padding)
        if frame2 is not None:
            I2 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 256.
            frame2 = F.pad(I2, padding)
        return frame1, frame2
    
    def cal_ssim(self, afterpre_frame1, afterpre_frame2):
        I0_small = F.interpolate(afterpre_frame1, (32, 32), mode='nearest-exact')
        I1_small = F.interpolate(afterpre_frame2, (32, 32), mode='nearest-exact')
        ssim = ssim_matlab(I0_small[:, :3].cpu(), I1_small[:, :3].cpu())
        return ssim

    def build_read_buffer(self, read_buffer, videogen):
        try:
            for frame in videogen:
                if not self.img is None:
                    frame = cv2.imread(os.path.join(self.img, frame))[:, :, ::-1].copy()
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(self, I0, I1, n):
        middle = self.model.inference(I0, I1, 1.0)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n//2)
        second_half = self.make_inference(middle, I1, n=n//2)

        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def infer(self):
        videogen = []
        for f in os.listdir(self.img):
            if 'png' in f:
                videogen.append(f)
        videogen = videogen[0:self.maxLen]
        tot_frame = len(videogen)
        videogen.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
        lastframe = cv2.imread(os.path.join(self.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()

        videogen = videogen[1:]
        h, w, _ = lastframe.shape
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        tmp = max(32, int(32 / 1.0))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=tot_frame)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_read_buffer, (read_buffer, videogen))

        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 256.
        I1 = F.pad(I1, padding)
        temp = None # save lastframe when processing static frame
        # inference
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1.to(self.device)
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 256.

            I1 = F.pad(I1, padding)
            I0_small = F.interpolate(I0, (32, 32), mode='nearest-exact')
            I1_small = F.interpolate(I1, (32, 32), mode='nearest-exact')

            # Fix: 
            # This function is not deterministic. A solution is required here.
            ssim = ssim_matlab(I0_small[:, :3].cpu(), I1_small[:, :3].cpu())
            break_flag = False
            if ssim > 0.996:        
                frame = read_buffer.get() # read a new frame
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 256.
                I1 = F.pad(I1, padding)
                I1 = self.model.inference(I0, I1, 1.0)
                I1_small = F.interpolate(I1, (32, 32), mode='nearest-exact').to(self.device)
                # Fix: 
                # This function is not deterministic. A solution is required here.
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 256).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
             
            if ssim < 0.2:
                output = []
                for i in range((2 ** self.exp) - 1):
                    output.append(I0)
            else:
                output = self.make_inference(I0, I1, 2**self.exp-1) if self.exp else []

            self.saveImg(lastframe)
  
            for mid in output:
                mid = (((mid[0] * 256.).byte().cpu().numpy().transpose(1, 2, 0)))
                self.saveImg(mid[:h, :w])

            pbar.update(1)
            lastframe = frame

            if break_flag:
                break

        self.saveImg(lastframe)
        pbar.update(1)
        pbar.close()
