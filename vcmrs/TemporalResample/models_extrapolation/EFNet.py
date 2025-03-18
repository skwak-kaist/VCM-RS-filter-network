import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from e2evc.Utils import utils_nn
import numpy as np
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backwarp_tenGrid = {}

def _sigmoid_stable(x):
   x2 = x*x
   x3 = x2*x
   y = 1/(2-x+x2*0.5-x3*0.16666667)
   y2 = 1-1/(2+x+x2*0.5+x3*0.16666667)
   return torch.where(x<0, y, y2)

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()
def warp(tenInput, tenFlow):

    # Fix:
    # torch linspace is not stable on different platform. The implementation is switched to numpy. However
    # a correct fix is required here.

    #tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device='cpu').view(
    #    1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1).to('cpu')
    #tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device='cpu').view(
    #    1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]).to('cpu')

    tenHorizontal = torch.tensor(np.linspace(-1.0, 1.0, tenFlow.shape[3])).float().view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1).to('cpu')
    tenVertical = torch.tensor(np.linspace(-1.0, 1.0, tenFlow.shape[2])).float().view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]).to('cpu')

    backwarp_tenGrid_tmp = torch.cat(
        [tenHorizontal, tenVertical], 1).to('cpu')

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :].to('cpu') / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :].to('cpu') / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid_tmp + tenFlow).permute(0, 2, 3, 1).to(device)
    # Fix:
    # this fuction is non-deterministic on cuda. A better solution is reqiured.
    aa = torch.nn.functional.grid_sample(input=tenInput.cpu(), grid=g.cpu(), mode="bilinear", align_corners=True,
                                         padding_mode='border')
    aa = aa.to(tenInput.device)

    return aa

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        utils_nn.IntConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def custom_bilinear_interpolation(x, scale_factor):
    b, n, h, w = x.shape
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    y = torch.arange(0, new_h, dtype=torch.float32, device=x.device)
    x_grid = torch.arange(0, new_w, dtype=torch.float32, device=x.device)

    y = (y + 0.5) / scale_factor - 0.5
    x_grid = (x_grid + 0.5) / scale_factor - 0.5

    y_grid, x_grid = torch.meshgrid(y, x_grid, indexing='ij')

    x0 = torch.floor(x_grid).long()
    x1 = x0 + 1
    y0 = torch.floor(y_grid).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, w - 1)
    x1 = torch.clamp(x1, 0, w - 1)
    y0 = torch.clamp(y0, 0, h - 1)
    y1 = torch.clamp(y1, 0, h - 1)

    dx = x_grid - x0.float()
    dy = y_grid - y0.float()
    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    wa = wa.unsqueeze(0).unsqueeze(0)
    wb = wb.unsqueeze(0).unsqueeze(0)
    wc = wc.unsqueeze(0).unsqueeze(0)
    wd = wd.unsqueeze(0).unsqueeze(0)

    output = (
            wa * x[:, :, y0, x0] +
            wb * x[:, :, y1, x0] +
            wc * x[:, :, y0, x1] +
            wd * x[:, :, y1, x1]
    )

    return output


class MPDWConv(nn.Module):
    """multi-scale-spatial-DWConv"""

    def __init__(self, embed_dims, dw_dilation=[1, 2, 3, 4], channel_split=[2, 2, 2, 2]):
        super(MPDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_3 = int(self.split_ratio[3] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2 - self.embed_dims_3
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 4
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 4
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = utils_nn.IntConv2d(
            in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=3,
            padding=1,
            groups=self.embed_dims, stride=1, dilation=1,
        )
        # DW conv 1
        self.DW_conv1 = utils_nn.IntConv2d(
            in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=3,
            padding=3,
            groups=self.embed_dims_1, stride=1, dilation=3,
        )
        # DW conv 2
        self.DW_conv2 = utils_nn.IntConv2d(
            in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=5,
            padding=6,
            groups=self.embed_dims_2, stride=1, dilation=3,
        )

        # DW conv 3
        self.DW_conv3 = utils_nn.IntConv2d(
            in_channels=self.embed_dims_3, out_channels=self.embed_dims_3, kernel_size=7,
            padding=9,
            groups=self.embed_dims_3, stride=1, dilation=3,
        )

        # a channel convolution
        self.PW_conv = utils_nn.IntConv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims_0+self.embed_dims_1:self.embed_dims_0+self.embed_dims_1+self.embed_dims_2, ...])
        x_3 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_3:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2, x_3], dim=1)
        x = self.PW_conv(x)
        return x

class MPN(nn.Module):
    """multi perspective network."""

    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3, 4], attn_channel_split=[2, 2, 2, 2]):
        super(MPN, self).__init__()
        self.embed_dims = embed_dims
        self.gate = utils_nn.IntConv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MPDWConv(
            embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = utils_nn.IntConv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

    def forward(self, x):
        shortcut = x
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        g = g*_sigmoid_stable(g)
        v = v*_sigmoid_stable(v)
        x = self.proj_2(g * v)
        x = shortcut + x
        return x

class EFBlock(nn.Module):
    def __init__(self, in_planes, num_feature, multiRF):
        super(EFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature//2, 3, 2, 1),
            conv(num_feature//2, num_feature, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature//4)

        self.conv1 = nn.Sequential(
            conv(in_planes, 8, 3, 2, 1),
        )
        # self.multiRF = MPN(embed_dims=8)
        self.multiRF = multiRF
        self.convblock1 = nn.Sequential(
            conv(8, 8),
        )
        self.lastconv = utils_nn.IntTransposedConv2d(num_feature//4 + 8, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = custom_bilinear_interpolation(x, scale_factor=1. / scale)
            flow = custom_bilinear_interpolation(flow, scale_factor=1. / scale) * 1. / scale
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = custom_bilinear_interpolation(x2, scale_factor=scale * 2)

        x3 = self.conv1(torch.cat((x0,flow0), 1))
        x3_2 = self.multiRF(x3)
        x4 = self.convblock1(x3_2)

        tmp = self.lastconv(torch.cat((x2, x4), dim=1))
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask



class EFNet(nn.Module):
    def __init__(self):
        super(EFNet, self).__init__()
        self.multiRF1 = MPN(embed_dims=8)
        self.multiRF2 = MPN(embed_dims=8)
        self.multiRF3 = MPN(embed_dims=8)
        self.block0 = EFBlock(13+4, num_feature=160, multiRF=self.multiRF1)
        self.block1 = EFBlock(13+4, num_feature=160, multiRF=self.multiRF1)
        self.block2 = EFBlock(13+4, num_feature=160, multiRF=self.multiRF1)
        self.block3 = EFBlock(13+4, num_feature=80, multiRF=self.multiRF2)
        self.block4 = EFBlock(13+4, num_feature=80, multiRF=self.multiRF2)
        self.block5 = EFBlock(13+4, num_feature=80, multiRF=self.multiRF2)
        self.block6 = EFBlock(13+4, num_feature=44, multiRF=self.multiRF3)
        self.block7 = EFBlock(13+4, num_feature=44, multiRF=self.multiRF3)
        self.block8 = EFBlock(13+4, num_feature=44, multiRF=self.multiRF3)

    def forward(self, x, scale, training=False):
        batch_size, _, height, width = x.shape

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        warped_img0 = img0
        warped_img1 = img1
        flow = Variable(torch.zeros(batch_size, 4, height, width)).to(device)
        mask = Variable(torch.zeros(batch_size, 1, height, width)).to(device)

        stu = [self.block0, self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7,
               self.block8]

        for i in range(9):
            flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                    scale=scale[i])
            flow = flow + flow_d
            mask = mask + mask_d
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        mask_last = _sigmoid_stable(mask)
        merged_last = (warped_img0, warped_img1)

        merged_final = merged_last[0] * mask_last + merged_last[1] * (1 - mask_last)
        merged_final = torch.clamp(merged_final, 0, 1)
        return merged_final
