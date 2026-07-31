import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.nn import init

import random 
import os     
import torch   


from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc




def set_seed(seed):
    """设置所有可能用到的随机种子以确保可复现性"""
    random.seed(seed) 
    np.random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

seed = 42  
set_seed(seed)



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class uc(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor):
        super(uc, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(Down, self).__init__()


        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,  
            stride=2,  
            padding=1,  
            bias=False  
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class SpatialAttentionAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        weights = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return x * weights  


class Attention_block(nn.Module):
    def __init__(self,  F_l, F_int):
        super(Attention_block, self).__init__()
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(F_int)
        # )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.sa = SpatialAttentionAG()
        # self.channel_adjust = nn.Conv2d(F_l * 2, F_l, kernel_size=1)
        self.conv3x31 = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True)
        )
        self.conv3x32 = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(F_l, F_l, kernel_size=1)

    def forward(self, x):
        x1 = self.W_x(x)
        psi1 = self.psi(x1)
        out1 = psi1 * x
        out2 = self.conv3x31(x)
        out3 = self.sa(out2)
        out6 = self.conv3x32(out3 + out1)
        return out6



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)





class HighPrecisionSobelEdgeConv(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, 3,
                                padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, 3,
                                padding=1, groups=channels, bias=False)

        # 初始化水平Sobel核(检测垂直边缘)
        sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).repeat(channels, 1, 1, 1)

        # 初始化垂直Sobel核(检测水平边缘)
        sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).repeat(channels, 1, 1, 1)

        self.conv_x.weight.data.copy_(sobel_kernel_x)
        self.conv_y.weight.data.copy_(sobel_kernel_y)
        self.conv_x.weight.requires_grad = True  
        self.conv_y.weight.requires_grad = True  

        self.alpha = nn.Parameter(torch.tensor(0.5))  

    def forward(self, x):

        edge_x = torch.abs(self.conv_x(x))
        edge_y = torch.abs(self.conv_y(x))

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  
        return x + self.alpha * x * torch.sigmoid(edge)


class MEU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        mid = max(16, in_channels // 2)


        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        self.up15x = self._make_level(mid, 1.5)
        self.up2x = self._make_level(mid, 2.0)

        self.attn = nn.Sequential(
            nn.Conv2d(mid * 2 + in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid * 2 + in_channels, 1),
            nn.Sigmoid()
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 2 + in_channels, mid * 2, 3, padding=1),
            nn.BatchNorm2d(mid * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )


        self.res_weight = nn.Parameter(torch.tensor(0.5))

    def _make_level(self, channels, scale):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            HighPrecisionSobelEdgeConv(channels)
        )

    def forward(self, x):
        orig_size = x.shape[2:]
        x_base = self.base_conv(x)


        x15 = self.up15x(x_base)
        x2 = self.up2x(x_base)

        x15 = F.interpolate(x15, size=orig_size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=orig_size, mode='bilinear', align_corners=True)

        concat_features = torch.cat([x15, x2, x], dim=1)
        attn_weights = self.attn(concat_features)
        weighted_features = concat_features * attn_weights


        fused = self.fusion(weighted_features)


        return x + self.res_weight * fused



class CAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.meu = MEU(in_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()  
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x31 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x32 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x33 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_planes=in_channels * 2)
        # self.cb = conv_block(ch_in=in_channels, ch_out=in_channels)
    def forward(self, x):
        y = x

        x1 = self.conv3x33(x)
        meu_feat = self.meu(x1)
        fused = torch.cat([x, meu_feat], dim=1)
        attn = self.ca(fused) * fused
        attn1 = self.conv3x31(attn)
        attn2 = self.conv3x32(attn1)


        # edge_weight = edge_maps.mean(dim=1, keepdim=True)  # [B,1,H,W]

        combined = attn2 + y  

        out = self.output_conv(combined)
        return out

class Attention41(nn.Module):
    def __init__(self, in_channels):
        super(Attention41, self).__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        fused = self.psi(x)
        out1 = x * fused
        out2 = 1 - fused
        out2 = x * out2
        out = self.conv1(out2 + out1)
        return out


class EnhancedBoundaryAttentionP2P3(nn.Module):
    def __init__(self, channels_p1, channels_p3, mid_channels=320):
        super(EnhancedBoundaryAttentionP2P3, self).__init__()
        self.channels_p1 = channels_p1
        self.channels_p3 = channels_p3
        self.mid_channels = mid_channels


        self.p1_adjust = nn.Sequential(
            nn.Conv2d(channels_p1, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.p3_adjust = nn.Sequential(
            nn.Conv2d(channels_p3, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )


        self.edge_conv_p1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=1)
        )
        self.edge_conv_p3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=1)
        )

        self.q_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.out_proj_p2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.out_proj_p3 = nn.Conv2d(mid_channels, channels_p3, kernel_size=1)


        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.AvgPool2d(kernel_size=2, stride=2)


        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, p1, p3):

        p1_adj = self.p1_adjust(p1)
        p3_adj = self.p3_adjust(p3)

        # 获取输入尺寸
        B, C, H_p3, W_p3 = p3_adj.shape
        _, _, H_p1, W_p1 = p1_adj.shape


        p1_up4 = self.upsample4(p1_adj)


        edge_p1 = torch.sigmoid(self.edge_conv_p1(p1_up4))
        edge_p3 = torch.sigmoid(self.edge_conv_p3(p3_adj))


        p1_modulated = p1_up4 * edge_p3 * self.alpha
        p3_modulated = p3_adj * edge_p1 * (1 - self.alpha)

        # 计算注意力
        q_p3 = self.q_proj(p3_modulated).view(B, C, -1).permute(0, 2, 1)
        k_p3 = self.k_proj(p1_modulated).view(B, C, -1)
        v_p3 = self.v_proj(p1_modulated).view(B, C, -1).permute(0, 2, 1)

        attn_weights_p3 = torch.bmm(q_p3, k_p3) / (C ** 0.5)
        attn_weights_p3 = F.softmax(attn_weights_p3, dim=-1)

        attended_p3 = torch.bmm(attn_weights_p3, v_p3)
        attended_p3 = attended_p3.permute(0, 2, 1).view(B, C, H_p3, W_p3)


        p3_out = self.out_proj_p3(p3_adj + attended_p3)

        H_p2, W_p2 = H_p3 // 2, W_p3 // 2

        p3_down = self.downsample2(p3_adj)
        p1_up2 = self.upsample2(p1_adj)


        edge_p3_down = self.downsample2(edge_p3)
        edge_p1_up2 = torch.sigmoid(self.edge_conv_p1(p1_up2))


        p3_modulated_p2 = p3_down * edge_p1_up2 * self.beta
        p1_modulated_p2 = p1_up2 * edge_p3_down * (1 - self.beta)

        # 计算注意力
        q_p2 = self.q_proj(p1_modulated_p2).view(B, C, -1).permute(0, 2, 1)
        k_p2 = self.k_proj(p3_modulated_p2).view(B, C, -1)
        v_p2 = self.v_proj(p3_modulated_p2).view(B, C, -1).permute(0, 2, 1)

        attn_weights_p2 = torch.bmm(q_p2, k_p2) / (C ** 0.5)
        attn_weights_p2 = F.softmax(attn_weights_p2, dim=-1)

        attended_p2 = torch.bmm(attn_weights_p2, v_p2)
        attended_p2 = attended_p2.permute(0, 2, 1).view(B, C, H_p2, W_p2)


        p2_out = self.out_proj_p2(p1_up2 + attended_p2)

        return p2_out, p3_out



class CASCADE(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CASCADE, self).__init__()

        # 特征初始化层
        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)  # 512-512
        # x4
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        # x3
        self.AG3 = Attention_block(F_l=channels[1], F_int=channels[2])
        self.CAM3 = CAM(in_channels=channels[1])

        # x2
        self.AG2 = Attention_block(F_l=channels[2], F_int=channels[3])
        self.CAM2 = CAM(in_channels=channels[2])

        # x1
        self.AG1 = Attention_block(F_l=channels[3], F_int=32)
        self.CAM1 = CAM(in_channels=channels[3])

        self.uc3 = uc(ch_in=channels[0], ch_out=channels[3], scale_factor=8)
        self.uc43 = uc(ch_in=channels[0], ch_out=channels[1], scale_factor=2)

        self.DOWN = Down(in_channels=channels[2], out_channels=channels[1])
        self.jh23 = EnhancedBoundaryAttentionP2P3(channels_p1=channels[0], channels_p3=channels[2])

        self.attention41 = Attention41(in_channels=channels[3] * 2)


    def forward(self, x, skips):
        d40 = self.Conv_1x1(x)
        d4 = self.ConvBlock4(d40)
        d4to1 = self.uc3(d4)
        x3 = self.AG3(x=skips[0])
        d31 = self.CAM3(x3)
        x2 = self.AG2(x=skips[1])
        d21 = self.CAM2(x2)
        x1 = self.AG1(x=skips[2])
        d11 = self.CAM1(x1)
        d12 = torch.cat([d11, d4to1], dim=1)
        d1 = self.attention41(d12)
        d32, d22 = self.jh23(d4, d21)
        d3 = d31
        d2 = d22 + d21
        d4 = d4

        return d4, d3, d2, d1

