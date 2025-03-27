import os
import torch
import torch.nn as nn
import tempfile
import glob
import numpy as np
import math
import subprocess
from PIL import Image
from time import perf_counter
from pathlib import Path

import cv2
import vcmrs
from vcmrs.JointFilter import unet, selection_algorithm
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink, makedirs

from e2evc.Utils import utils_nn

class filter(Component): 
    def __init__(self, ctx, mode):
        super().__init__(ctx)
        self.mode = mode
    
    def process(self, input_fname, output_fname, item, ctx):
        if self.mode == 'pre':
            is_yuv = item.args.JointFilterPreDomain == 'YUV'
            height, width = item.args.SourceHeight, item.args.SourceWidth
            if item.args.JointFilterPreModel == 'Identity':
                vcmrs.log(f"[Joint Filter] Pre filter off")
                self.process_identity(input_fname, output_fname, item, ctx)
                return
            vcmrs.log(f"[Joint Filter] Pre filter on")
            net = globals()[item.args.JointFilterPreModel]()
            
        elif self.mode == 'post':
            is_yuv = item.args.JointFilterPostDomain == 'YUV'
            item.args.InputBitDepth = 10
            post_flag = item.args.JointFilterPostModel != 'Identity'
            if post_flag:
                post_flag = self._hijack_bdt_flag(item) # hijack BitDepthTruncation flag
                #post_flag = True # 일단 무조건 켜는 방안으로 hard coding
            height, width, _ = item.video_info.resolution
            if not post_flag:
                vcmrs.log(f"[Joint Filter] Post filter off")
                self.process_identity(input_fname, output_fname, item, ctx)
                return
            vcmrs.log(f"[Joint Filter] Post filter on")
            net = globals()[item.args.JointFilterPostModel]()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        ckpt_selection = getattr(selection_algorithm, item.args.JointFilterSelectionAlgorithm)
        try:
            ckpt_path = ckpt_selection(item.args.quality) / f"{self.mode}filter.pth"
        except AttributeError:
            ckpt_path = ckpt_selection(None) / f"{self.mode}filter.pth"

        fn = get_filter_fn(net, ckpt_path, patch_size=item.args.JointFilterPatchSize)
        vcmrs.log(f"================= {self.mode}-filtering with chekpoint on {ckpt_path} ====================")

        if item._is_dir_video:
            # video data in a directory
            fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
            for idx, fname in enumerate(fnames):
                output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
                if os.path.isfile(output_frame_fname): os.remove(output_frame_fname)
                os.makedirs(os.path.dirname(output_frame_fname), exist_ok=True)
                fn(Image.open(fname)).save(output_frame_fname) # type: ignore

        elif item._is_yuv_video:
            if is_yuv:
                makedirs(os.path.dirname(output_fname))
                output_writer = open(output_fname, 'wb')
                for yuv_frame in read_yuv_frames(input_fname, height, width, item.args.InputBitDepth, item.args.InputChromaFormat, normalize=True):
                    save_yuv_frame(output_writer, fn(yuv_frame), item.args.InputBitDepth, item.args.InputChromaFormat, denormalize=True)
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    yuv_to_png(input_fname, temp_dir, width, height, item.args.InputBitDepth, start_frame_number=1)
                    for png in sorted(temp_dir.glob("*.png")):
                        fn(Image.open(png)).save(png) # type: ignore
                    makedirs(os.path.dirname(output_fname))
                    png_to_yuv(temp_dir, output_fname, item.args.InputBitDepth)

        else:
            if os.path.isfile(output_fname): os.remove(output_fname)
            os.makedirs(os.path.dirname(output_fname), exist_ok=True)
            fn(Image.open(input_fname)).save(output_fname) # type: ignore
            
    def process_identity(self, input_fname, output_fname, item, ctx):
        # the default implementation is a bypass component

        #if item._is_dir_video:
        if os.path.isdir(input_fname): #item._is_dir_video:
            # video data in a directory
            fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
            for idx, fname in enumerate(fnames):
                output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
                if os.path.isfile(output_frame_fname): os.remove(output_frame_fname)
                makedirs(os.path.dirname(output_frame_fname))
                enforce_symlink(fname, output_frame_fname)
        else:
            # image data or video in yuv format
            if os.path.isfile(output_fname): os.remove(output_fname)
            makedirs(os.path.dirname(output_fname))
            enforce_symlink(input_fname, output_fname)
            
    def _hijack_bdt_flag(self, item):
        param_data = item.parameters['BitDepthTruncation']
        param_data_bytearray = bytearray(param_data)
        if param_data_bytearray[0]: # `bit_depth_shift_flag` is on
            param_data_bytearray[0] = 0 # Turn off `bit_depth_shift_flag`
            item.parameters['BitDepthTruncation'] = bytes(param_data_bytearray)
            return True
        else:
            return False


def yuv_to_png(yuv_path, png_path, width, height, input_bit_depth, start_frame_number=1):
    if png_path.exists():
        assert png_path.is_dir()
    else:
        png_path.mkdir(parents=True)
        
    pix_fmt = "yuv420p" if input_bit_depth == 8 else "yuv420p10le"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-s",
        f"{width}x{height}", "-pix_fmt", pix_fmt, "-i", str(yuv_path),
        "-vsync", "1",  "-pix_fmt", "rgb24",
        "-start_number", str(start_frame_number), str(png_path / "%06d.png")
    ]
    run_cmd(cmd)


def png_to_yuv(png_path, yuv_path, output_bit_depth):
    pix_fmt = "yuv420p" if output_bit_depth == 8 else "yuv420p10le"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-pix_fmt", "rgb24", "-i", str(png_path / "%06d.png"),
        "-f", "rawvideo", "-pix_fmt", pix_fmt, str(yuv_path)
    ]
    run_cmd(cmd)


def run_cmd(cmd):
    subprocess.run(
        cmd, check=True,
        stdin=subprocess.PIPE,
    ) 


def read_yuv_frames(yuv_path, height, width, bit_depth, input_chroma_format, normalize=True):
    y_size = width * height
    bytes_per_pixel = 2 if bit_depth == 10 else 1
    
    if input_chroma_format == "420":
        uv_size = (width // 2) * (height // 2)
        uv_shape = (height // 2, width // 2)
        upsample_size = (width, height)  # Target resolution for U and V planes
    elif input_chroma_format == "422":
        uv_size = (width // 2) * height
        uv_shape = (height, width // 2)
        upsample_size = (width, height)  # Target resolution for U and V planes
    elif input_chroma_format == "444":
        uv_size = width * height
        uv_shape = (height, width)
        upsample_size = None  # No upsampling needed for YUV 4:4:4
    else:
        raise ValueError("Unsupported chroma format: {}".format(input_chroma_format))

    with open(yuv_path, 'rb') as yuv_file:
        while True:
            # Read Y component
            y_frame = yuv_file.read(y_size * bytes_per_pixel)
            if len(y_frame) < y_size * bytes_per_pixel:
                break  # End of file
            
            # Read U component
            u_frame = yuv_file.read(uv_size * bytes_per_pixel)
            if len(u_frame) < uv_size * bytes_per_pixel:
                break  # End of file
            
            # Read V component
            v_frame = yuv_file.read(uv_size * bytes_per_pixel)
            if len(v_frame) < uv_size * bytes_per_pixel:
                break  # End of file

            # Convert to NumPy arrays
            y_plane = np.frombuffer(y_frame, dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape((height, width))
            u_plane = np.frombuffer(u_frame, dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)
            v_plane = np.frombuffer(v_frame, dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)

            # Upsample U and V to match Y resolution using cv2.resize
            if input_chroma_format != "444":
                u_plane = cv2.resize(u_plane, upsample_size, interpolation=cv2.INTER_LINEAR)
                v_plane = cv2.resize(v_plane, upsample_size, interpolation=cv2.INTER_LINEAR)
                
            if normalize:
                y_plane = y_plane.astype(np.float32)
                u_plane = u_plane.astype(np.float32)
                v_plane = v_plane.astype(np.float32)
                if bit_depth == 10:
                    y_plane = (y_plane - 64) / (940 - 64)   # Scale Y from [64, 940] to [0, 1]
                    u_plane = (u_plane - 64) / (960 - 64)   # Scale U from [64, 960] to [0, 1]
                    v_plane = (v_plane - 64) / (960 - 64)   # Scale V from [64, 960] to [0, 1]
                elif bit_depth == 8:
                    y_plane = (y_plane - 16) / (235 - 16)   # Scale Y from [16, 235] to [0, 1]
                    u_plane = (u_plane - 16) / (240 - 16)   # Scale U from [16, 240] to [0, 1]
                    v_plane = (v_plane - 16) / (240 - 16)   # Scale V from [16, 240] to [0, 1]

            # Combine Y, U, and V into a 3-channel YUV frame
            yuv_frame = np.stack([y_plane, u_plane, v_plane], axis=-1)

            # Yield the YUV frame
            yield yuv_frame
            

def save_yuv_frame(output_writer, yuv_frame, bit_depth, input_chroma_format, denormalize=True):
    bytes_per_pixel = 2 if bit_depth == 10 else 1
    yuv_frame = yuv_frame.copy()
    
    if denormalize:
        if bit_depth == 10:
            yuv_frame[..., 0] = yuv_frame[..., 0] * (940 - 64) + 64
            yuv_frame[..., 1] = yuv_frame[..., 1] * (960 - 64) + 64
            yuv_frame[..., 2] = yuv_frame[..., 2] * (960 - 64) + 64
        elif bit_depth == 8:
            yuv_frame[..., 0] = yuv_frame[..., 0] * (235 - 16) + 16
            yuv_frame[..., 1] = yuv_frame[..., 1] * (240 - 16) + 16
            yuv_frame[..., 2] = yuv_frame[..., 2] * (240 - 16) + 16
    
    yuv_frame = yuv_frame.astype(np.uint16 if bytes_per_pixel == 2 else np.uint8)
    y_plane = yuv_frame[..., 0]
    u_plane = yuv_frame[..., 1]
    v_plane = yuv_frame[..., 2]
    
    height, width = y_plane.shape
    
    # Downsample U and V planes if needed
    if input_chroma_format == "420":
        u_plane = cv2.resize(u_plane, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        v_plane = cv2.resize(v_plane, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    elif input_chroma_format == "422":
        u_plane = cv2.resize(u_plane, (width // 2, height), interpolation=cv2.INTER_LINEAR)
        v_plane = cv2.resize(v_plane, (width // 2, height), interpolation=cv2.INTER_LINEAR)
    # No need to downsample for YUV444

    # Convert planes to bytes and write to file
    output_writer.write(y_plane.tobytes())
    output_writer.write(u_plane.tobytes())
    output_writer.write(v_plane.tobytes())

    
F_IN_OUT = [(3, 16), (16, 32), (32, 64), (64, 32), (32, 16)]

class FilterV8(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet_encoder = unet.Encoder(in_channels=in_channels)
        self.unet_decoder = unet.Decoder(out_channels=out_channels)
        self.conv_1x1_a = nn.Sequential(
            #nn.Conv2d(in_channels, 1, kernel_size=1, stride=1),
            utils_nn.IntConv2d(in_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv_1x1_b = nn.Sequential(
            #nn.Conv2d(1, out_channels, kernel_size=1, stride=1),
            utils_nn.IntConv2d(1, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_unet_5, x_unet_4, x_unet_3, x_unet_2, x_unet_1 = self.unet_encoder(x)
        x_conv_1x1 = self.conv_1x1_a(x)
        x_unet = self.unet_decoder(x_unet_5, x_unet_4, x_unet_3, x_unet_2, x_unet_1)
        x_conv_1x1 = self.conv_1x1_b(x_conv_1x1)
        return self.sigmoid(x_unet + x_conv_1x1)


def get_filter_fn(net, checkpoint_path, patch_size=-1, use_cuda=True, decoding_time=False):
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net = net.cuda() if use_cuda else net
    net.eval()

    def fn(image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            if decoding_time:
                image_filtered, dt = fn(image)
                return Image.fromarray(image_filtered), dt
            image_filtered = fn(image)
            return Image.fromarray(image_filtered)
        if isinstance(image, np.ndarray):
            dtype = image.dtype
            image = image.transpose(2, 0, 1)
            if dtype == np.uint8:
                image = image.astype('float32') / 255.
            image = torch.from_numpy(image)
            if decoding_time:
                image_filtered, dt = fn(image)
                image_filtered = image_filtered.cpu().numpy()
                if dtype == np.uint8:
                    image_filtered = (image_filtered * 255.).round().astype('uint8')
                image_filtered = image_filtered.transpose(1, 2, 0)
                return image_filtered, dt
            image_filtered = fn(image)
            image_filtered = image_filtered.cpu().numpy()
            if dtype == np.uint8:
                image_filtered = (image_filtered * 255.).round().astype('uint8')
            image_filtered = image_filtered.transpose(1, 2, 0)
            return image_filtered

        # torch.Tensor, 0 ~ 1. non-batched
        h, w = image.shape[-2:]
        ph, pw = patch_size, patch_size
        image = image.cuda() if use_cuda else image

        def _fn(image):
            if patch_size == -1:
                with torch.no_grad():
                    return net(image)
            else:
                rows = []
                for i in range(math.ceil(h / ph)):
                    row = []
                    for j in range(math.ceil(w / pw)):
                        patch = image[..., i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        with torch.no_grad():
                            patch_filtered = net(patch)
                        row.append(patch_filtered)
                    row = torch.cat(row, dim=-1)
                    rows.append(row)
                return torch.cat(rows, dim=-2)

        st = perf_counter()
        image_filtered = _fn(image[None, ...])[0]
        dt = perf_counter() - st
        if decoding_time:
            return image_filtered, dt
        return image_filtered
    return fn