
from PIL import Image
import numpy as np
#from skimage import color  # scikit-image>=0.21.0
import torch
import torch.nn.functional as F
#from IPython import embed

from vcmrs.ROI.roi_retargeting import roi_interpolation

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np


# PIL.Image.NEAREST=0
# PIL.Image.LANCZOS=1
# PIL.Image.BILINEAR = 2
# PIL.Image.BICUBIC = 3

def resize_img(img, HW=(256,256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def resize_onecomp(inp_image, HW=(256,256) ):

  if (inp_image.shape[0] == HW[0]) and (inp_image.shape[1] == HW[1]):
    return inp_image
    
  out_image = np.zeros( HW, dtype = inp_image.dtype)
  roi_interpolation.BilinearInterpolationNumbaIntOneComp(out_image, inp_image )
  return out_image
      

#def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
#    # return original size L and resized L as torch Tensors
#    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
#    
#    img_lab_orig = color.rgb2lab(img_rgb_orig)
#    img_lab_rs = color.rgb2lab(img_rgb_rs)
#
#    img_l_orig = img_lab_orig[:,:,0]
#    img_l_rs = img_lab_rs[:,:,0]
#
#    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
#    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]
#
#    return (tens_orig_l, tens_rs_l)

def preprocess_img_yuv(img_y_orig, bit_depth, HW=(256,256)):
    img_y_rs = resize_onecomp(img_y_orig, HW)
    scale = (1<<(bit_depth))
    img_y_rs = img_y_rs.astype(np.float32) / scale      # 0..1   # -0.5...0.5
    tens_rs_y = torch.Tensor(img_y_rs)[None,None,:,:]

    return tens_rs_y


#def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
#    # tens_orig_l     1 x 1 x H_orig x W_orig
#    # out_ab          1 x 2 x H x W
#
#    HW_orig = tens_orig_l.shape[2:]
#    HW = out_ab.shape[2:]
#    
#    #(out_img*255).astype(np.uint8)
#
#    # call resize function if needed
#    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
#        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
#    else:
#        out_ab_orig = out_ab
#
#    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
#    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

#import random 

def postprocess_tens_yuv(img_u_orig, img_v_orig,   bit_depth, out_uv):
    # out_uv         1 x 2 x H x W
    out_uv = out_uv.data.cpu().numpy()
    out_u = out_uv[0,0,...]
    out_v = out_uv[0,1,...]
    
    out_u = out_u.astype(np.float32)
    out_v = out_v.astype(np.float32)

    scale = (1<<(bit_depth))
    max_value = (1<<(bit_depth))-1

    out_u = np.round(np.clip( out_u*scale, a_min=0, a_max = max_value)).astype(img_u_orig.dtype)
    out_v = np.round(np.clip( out_v*scale, a_min=0, a_max = max_value)).astype(img_v_orig.dtype)
    
    out_u_orig = resize_onecomp(out_u, img_u_orig.shape)
    out_v_orig = resize_onecomp(out_v, img_v_orig.shape)
    
    return out_u_orig, out_v_orig

    

    