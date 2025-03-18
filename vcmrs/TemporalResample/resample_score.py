# This file is covered by the license agreement found in the file "license.txt" in the root of this project.\
import os
import torch
import numpy as np
import random
import cv2
from torchvision import transforms
from vcmrs.TemporalResample.models.maniqa import MANIQA
from pathlib import Path
import vcmrs.TemporalResample.models as temporal_model
from vcmrs.Utils.io_utils import makedirs

torch.cuda.empty_cache()
temporal_model_dir = Path(temporal_model.__path__[0])

class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        d_name = sample['d_name']

        d_img = (d_img - self.mean) / self.var

        sample = {'d_img_org': d_img, 'd_name': d_name}
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_name = sample['d_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False    
        torch.backends.cudnn.deterministic = False


class _TemporalYUV_(torch.utils.data.Dataset):
    def __init__(self, y_img, h, w, transform, max_rows, max_cols, num_crops=20):
        super(_TemporalYUV_, self).__init__()

        self.new_h = 224
        self.new_w = 224
        if h<self.new_h: 
            y_img = cv2.resize(y_img, (w, 224))
            h = self.new_h
        if w<self.new_w:
            y_img = cv2.resize(y_img, (224, h))
            w = self.new_w

        self.img = np.array(y_img).astype('float32') / 1023.
        self.img = np.transpose(self.img, (2, 0, 1))
        self.transform = transform
        
        img_patches = []
        rest_w = (w- max_cols*self.new_w)//2
        rest_h = (h- max_rows*self.new_h)//2
        # Generate patches by iterating through rows and columns
        for i in range(max_rows):
            for j in range(max_cols):
                top = i * self.new_h
                left = j * self.new_w
                patch = self.img[:, rest_h+top: rest_h+top + self.new_h, rest_w+left: rest_w+left + self.new_w]
                img_patches.append(patch)        
        
        self.img_patches = np.array(img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': 'yuv'}
        if self.transform:
            sample = self.transform(sample)
        return sample


class _TemporalAssesser_():
    def __init__(self,h, w):
        super(_TemporalAssesser_, self).__init__()
        setup_seed(20)
        cpu_num = 1
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num) 

        self.height = h
        self.width = w
        
        self.patch_size = 8
        self.img_size = 224
        max_crops = 20 

        max_rows = h // self.img_size
        if max_rows==0: max_rows=1
        max_cols = w // self.img_size
        if max_cols==0: max_cols=1
        self.max_rows = max_rows
        self.max_cols = max_cols
        
        # Adjust rows and columns to keep num_crops <= max_crops
        num_crops = max_rows * max_cols
        while num_crops > max_crops:
            if max_rows >= max_cols and max_rows > 1:
                max_rows -= 1
            elif max_cols > 1:
                max_cols -= 1
            num_crops = max_rows * max_cols
        self.num_crops = num_crops

        self.embed_dim = 768
        self.dim_mlp = 768
        self.num_heads = [4, 4]
        self.window_size = 4
        self.depths = [2, 2]
        self.num_outputs = 1
        self.num_tab = 2
        self.scale = 0.8
        self.model_path = f'{temporal_model_dir}/ckpt_koniq10k.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net =  MANIQA(embed_dim=self.embed_dim, num_outputs=self.num_outputs, dim_mlp=self.dim_mlp,
        patch_size=self.patch_size, img_size=self.img_size, window_size=self.window_size,
        depths=self.depths, num_heads=self.num_heads, num_tab=self.num_tab, scale=self.scale, device=self.device)

        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.net.to(self.device)     
        self.net.eval()

    def score_quantization(self, score, bitdepth):
        Q_score = np.round(score*(2**bitdepth - 1))
        return Q_score

    def process_yuvs_assessment(self, input_fname, input_chroma_format, input_bit_depth, q_bit_depth, frame_num, call_position='enc', path=None):
        height = self.height
        width = self.width
        input_bytes_per_pixel = 2 if input_bit_depth > 8 else 1
        input_dtype = np.uint16 if input_bytes_per_pixel == 2 else np.uint8
        if input_chroma_format == "420":
            y_size = width * height
            uv_size = (width // 2) * (height // 2)
            uv_shape = (height // 2, width // 2)
        elif input_chroma_format == "422":
            y_size = width * height
            uv_size = (width // 2) * height
            uv_shape = (height, width // 2)
        elif input_chroma_format == "444":
            y_size = width * height
            uv_size = width * height
            uv_shape = (height, width)
        else:
            raise ValueError("Unsupported chroma format: {}".format(input_chroma_format))   
    
        input_file = open(input_fname, 'rb')
        out_frame_idx = 0
        score_list = []
        while True:
            # read Y, U, V
            y_buffer = input_file.read(y_size * input_bytes_per_pixel)
            if len(y_buffer) == 0: break  # check if at the end of the file
            
            u_buffer = input_file.read(uv_size * input_bytes_per_pixel)
            v_buffer = input_file.read(uv_size * input_bytes_per_pixel)

            y_array = np.frombuffer(y_buffer, dtype=input_dtype).reshape((height, width))
            y_img = np.stack((y_array,)*3, axis=-1)

            avg_score = 0
            Img = _TemporalYUV_(y_img, height, width, transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]), max_rows=self.max_rows, max_cols=self.max_cols, num_crops=self.num_crops)
            for patch_idx in range(self.num_crops):
                patch_sample = Img.get_patch(patch_idx)
                patch = patch_sample['d_img_org'].to(self.device)
                patch = patch.unsqueeze(0)
                score = self.net(patch)
                avg_score += score                    
            avg_score = avg_score / self.num_crops
            if q_bit_depth == 0:
                avg_score = avg_score.item()
            else: 
                avg_score = self.score_quantization(avg_score.item(), q_bit_depth)
            score_list.append(avg_score)
            out_frame_idx += 1
            if frame_num:
                if out_frame_idx>=frame_num:
                    break  
        input_file.close()
        torch.cuda.empty_cache()
        if call_position == 'enc':
            return score_list             
        elif call_position == 'dec':
            if path != None:
                makedirs(f"{path}/score_{call_position}")
                file_name=f"{path}/score_{call_position}/score_list.txt"
                with open(file_name, 'w') as file:
                    for score in score_list:
                        file.write(f"{str(int(score))}\n")
        else:
            raise ValueError("Unsupported codec position: {}".format(call_position))


