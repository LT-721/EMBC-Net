import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.nn import init

import random  # Python内置随机库
import os      # 操作系统接口库
import torch   # PyTorch深度学习框架


from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc




def set_seed(seed):
    """设置所有可能用到的随机种子以确保可复现性"""
    random.seed(seed)  # Python内置的随机库
    np.random.seed(seed)  # Numpy库
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希随机化，影响Python的哈希基础设施
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果不设置这个选项，可能因为cudnn的优化而使得计算结果不可复现
    torch.backends.cudnn.benchmark = False  # 当网络输入数据维度或类型上变化不大时，设置为True可以增加运行效率

seed = 42  # 或者其他你喜欢的数字
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
        """
        2倍下采样模块
        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
        """
        super(Down, self).__init__()

        # 使用步长为2的卷积实现下采样
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,  # 3x3卷积核
            stride=2,  # 步长2实现下采样
            padding=1,  # 保持输出尺寸计算正确
            bias=False  # 后面有BN层，不需要bias
        )

        # 批标准化加速收敛
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, in_channels, H, W]
        返回:
            下采样后的张量 [batch_size, out_channels, H//2, W//2]
        """
        x = self.conv(x)
        x = self.bn(x)
        return x




# AG模块
class ChannelAttentionAG(nn.Module):
    """轻量化通道注意力（适配AG模块特征维度）"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x形状: [B, C, H, W]
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))  # [B, C]
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))  # [B, C]
        weights = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return x * weights  # 通道加权


class SpatialAttentionAG(nn.Module):
    """轻量化空间注意力（保持原始AG的1x1卷积风格）"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x形状: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        weights = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return x * weights  # 空间加权

# AG模块

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



class MultiBranchFusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # 分支1：扩张率1的深度可分离卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size - 1) // 2, dilation=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2：扩张率2的深度可分离卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=2 * (kernel_size - 1) // 2, dilation=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支3：扩张率4的深度可分离卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=4 * (kernel_size - 1) // 2, dilation=4, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支4：平均池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 输出融合后的特征
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 四个分支并行处理
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 特征相加融合
        out = b1 + b2 + b3 + b4
        return out


class HighPrecisionSobelEdgeConv(nn.Module):
    """高精度可学习Sobel卷积"""

    def __init__(self, channels):
        super().__init__()
        # 使用分组卷积保持通道独立性
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
        self.conv_x.weight.requires_grad = True  # 允许微调
        self.conv_y.weight.requires_grad = True  # 允许微调

        # 增强边缘响应的可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self, x):
        # 计算水平和垂直边缘
        edge_x = torch.abs(self.conv_x(x))
        edge_y = torch.abs(self.conv_y(x))
        # 合并边缘强度
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # 添加小常数防止数值不稳定
        # 使用更精细的边缘增强
        return x + self.alpha * x * torch.sigmoid(edge)


class MEU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 保持足够的通道数
        mid = max(16, in_channels // 2)

        # 基础卷积层 - 保持强大特征提取能力
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 仅保留1.5x和2x上采样分支
        self.up15x = self._make_level(mid, 1.5)
        self.up2x = self._make_level(mid, 2.0)

        # 增强注意力机制 (补偿去除3x的精度损失)
        self.attn = nn.Sequential(
            nn.Conv2d(mid * 2 + in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid * 2 + in_channels, 1),
            nn.Sigmoid()
        )

        # 改进的特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 2 + in_channels, mid * 2, 3, padding=1),
            nn.BatchNorm2d(mid * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

        # 可学习的残差权重
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

        # 并行计算两个分支
        x15 = self.up15x(x_base)
        x2 = self.up2x(x_base)

        # 统一降采样到原始尺寸
        x15 = F.interpolate(x15, size=orig_size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=orig_size, mode='bilinear', align_corners=True)

        # 合并特征并应用注意力机制
        concat_features = torch.cat([x15, x2, x], dim=1)
        attn_weights = self.attn(concat_features)
        weighted_features = concat_features * attn_weights

        # 最终融合
        fused = self.fusion(weighted_features)

        # 自适应残差连接
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
            nn.Sigmoid()  # 输出单通道注意力
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
        # 获取多尺度特征和边缘
        x1 = self.conv3x33(x)
        meu_feat = self.meu(x1)
        fused = torch.cat([x, meu_feat], dim=1)
        attn = self.ca(fused) * fused
        attn1 = self.conv3x31(attn)
        attn2 = self.conv3x32(attn1)

        # 边缘图融合（3通道->1通道）
        # edge_weight = edge_maps.mean(dim=1, keepdim=True)  # [B,1,H,W]

        combined = attn2 + y  # 特征相加 [B,C,H,W]
        # 残差连接
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

        # 通道调整卷积（添加BN和ReLU）
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

        # 边界图生成（使用更精细的边缘检测）
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

        # QKV投影层（添加更多非线性变换）
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

        # 输出投影层
        self.out_proj_p2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.out_proj_p3 = nn.Conv2d(mid_channels, channels_p3, kernel_size=1)

        # 上采样和下采样层
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 添加可学习权重平衡不同分辨率贡献
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, p1, p3):
        # 调整通道数
        p1_adj = self.p1_adjust(p1)
        p3_adj = self.p3_adjust(p3)

        # 获取输入尺寸
        B, C, H_p3, W_p3 = p3_adj.shape
        _, _, H_p1, W_p1 = p1_adj.shape

        # 将P1上采样4倍到P3的尺寸
        p1_up4 = self.upsample4(p1_adj)

        # 生成边界图
        edge_p1 = torch.sigmoid(self.edge_conv_p1(p1_up4))
        edge_p3 = torch.sigmoid(self.edge_conv_p3(p3_adj))

        # 使用边界图调制特征（添加可学习权重）
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

        # 残差连接和输出投影
        p3_out = self.out_proj_p3(p3_adj + attended_p3)

        # P2分支
        H_p2, W_p2 = H_p3 // 2, W_p3 // 2

        p3_down = self.downsample2(p3_adj)
        p1_up2 = self.upsample2(p1_adj)

        # 生成P2尺寸的边界图
        edge_p3_down = self.downsample2(edge_p3)
        edge_p1_up2 = torch.sigmoid(self.edge_conv_p1(p1_up2))

        # 调制特征
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

        # 残差连接和输出投影
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
        # 通道对齐 → 通道注意 → 空间注意 → 特征精炼, 增强后的 d4（尺寸 H/32×W/32×512）
        d40 = self.Conv_1x1(x)
        d4 = self.ConvBlock4(d40)
        d4to1 = self.uc3(d4)
        d4to3 = self.uc43(d4)
        x3 = self.AG3(x=skips[0])
        d31 = self.CAM3(x3)
        x2 = self.AG2(x=skips[1])
        d21 = self.CAM2(x2)
        x1 = self.AG1(x=skips[2])
        d11 = self.CAM1(x1)
        d12 = torch.cat([d11, d4to1], dim=1)
        d1 = self.attention41(d12)
        d32, d22 = self.jh23(d4, d21)
        d3 = d31 + d4to3
        d2 = d22 + d21
        d4 = d4

        return d4, d3, d2, d1

