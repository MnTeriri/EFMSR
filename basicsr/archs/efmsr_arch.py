import math

import torch
from einops import rearrange
from fvcore.nn import parameter_count_table
from thop import profile, clever_format
from torch import nn as nn, Tensor

from timm.layers import trunc_normal_, to_2tuple

from basicsr.archs.pfa import WindowAttention
from basicsr.utils.registry import ARCH_REGISTRY


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
        将输入的特征图 x 切分成若干不重叠的窗口（patches），每个窗口大小为 (window_size, window_size)，并把所有窗口展平成一个批次返回

    Args:
        x: 形状为 [B, H, W, C] 的输入张量
        window_size (int): 窗口大小

    Returns:
        windows: 形状为 [num_windows * B, window_size, window_size, C] 的窗口张量
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    r"""
        将切分成多个窗口的小块 windows，恢复成原始大小的特征图 x

    Args:
        windows: 形状为 [num_windows * B, window_size, window_size, C] 的窗口张量
        window_size (int): 窗口大小
        H (int): 图像高度
        W (int): 图像宽度

    Returns:
        x: 形状为 [B, H, W, C] 的重组后的张量
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Upsample(nn.Sequential):
    """
    上采样模块。

    Args:
        scale (int): 放大倍数。支持的倍数为 2 的幂（2^n）和 3。
        num_feat (int): 中间特征图的通道数。
    """

    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """
    一次性上采样模块（与 Upsample 的区别在于它始终只包含 1 个卷积和 1 个 PixelShuffle）。
    用于轻量级超分辨率模型，以节省参数量。

    Args:
        scale (int): 放大倍数。支持的倍数为 2 的幂（2^n）和 3。
        num_feat (int): 中间特征的通道数。
        out_channel (int)：输出通道数。
    """

    def __init__(self, scale: int, num_feat: int, out_channel: int):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * out_channel, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super().__init__(*m)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:Tensor of shape [B, H*W, C]
        """
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()  # [B, C, H, W] → [B, H*W, C]
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, H*W, C]
            x_size: 张量 x 的 H 和 W

        Returns: Tensor of shape [B, C, H, W]
        """
        H, W = x_size
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()  # [B, H*W, C] → [B, C, H, W]
        return x


class ChannelAttention(nn.Module):
    """
    通道注意力模块

    Args:
        num_feat (int): 输入特征图的通道数。
        squeeze_factor (int, optional): 通道压缩比例，用于减少中间隐藏层通道数，默认值为16。

    该模块通过全局平均池化捕捉通道全局信息，经过两层1x1卷积（中间通道数为 num_feat // squeeze_factor），
    最后用 sigmoid 产生通道注意力权重，对输入特征图进行加权。
    """

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns: Tensor of shape [B, C, H, W]
        """
        y = self.attention(x)
        return x * y


class ChannelAttentionBlock(nn.Module):
    """
    通道注意力块（Channel Attention Block）

    Args:
        num_feat (int): 输入和输出特征图的通道数。
        compress_ratio (int): 通道压缩比例，用于中间卷积层降低通道数，默认值为3。
        squeeze_factor (int): 通道注意力模块中的挤压因子，用于调整通道注意力机制的中间维度，默认值为30。

    该模块通过两层卷积和 GELU 激活进行通道变换，最后使用通道注意力（ChannelAttention）机制自适应调整通道权重。
    """

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns: Tensor of shape [B, C, H, W]
        """
        return self.cab(x)


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2).contiguous()
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).contiguous().reshape(new_h, new_w, num_heads, 2)
    return new_weight


class FreqChannelScaler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 可学习的局部滤波器参数 [1, 1, dim]
        self.weight = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, H, W, C]

        Returns: Tensor of shape [B, H, W, C]
        """
        return x * self.weight[None, :, :, :]  # 加 batch 维广播


class DynamicFilter(nn.Module):
    """
    R-IDFM: Real-Imag Dynamic Filtering Module
    """

    def __init__(
            self,
            dim: int,
            window_size: int = 16,
            expansion_ratio: int = 2,
            reweight_expansion_ratio: float = .25,
            num_filters: int = 4,
            bias: bool = False,
            weight_resize: bool = False,
            act1_layer=StarReLU,
            act2_layer=nn.Identity
    ):
        super().__init__()
        size = to_2tuple(window_size)
        self.size = int(size[0])
        self.filter_size = int(size[1] // 2 + 1)
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()

        self.reweight = Mlp(
            in_features=dim,
            hidden_features=int(dim * reweight_expansion_ratio),
            out_features=num_filters * self.med_channels
        )
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02)

        self.freq_channel_scaler_real = FreqChannelScaler(self.med_channels)
        self.freq_channel_scaler_imag = FreqChannelScaler(self.med_channels)

        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, H, W, C]

        Returns:
            Tensor of shape [B, H, W, C]
        """
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, H, W)
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, H, W, self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        x = x * weight

        temp1 = self.freq_channel_scaler_real(x.real)
        temp2 = self.freq_channel_scaler_imag(x.imag)

        x = torch.complex(temp1 - temp2, temp1 + temp2)  # 拼回复数

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """
    深度可分离卷积模块（Depthwise Convolution），用于增强局部感知能力。

    本模块适用于 Transformer 架构中的前馈网络（如 ConvFFN），通过将序列特征重构为图像形状，利用 2D 深度卷积提取局部空间特征。

    Args:
        hidden_features (int): 特征通道数，亦为卷积输入输出通道数。
        kernel_size (int): 卷积核大小（默认 5），越大感受野越广。

    结构:
        - 将序列特征 reshape 为 [B, C, H, W]
        - 使用深度可分离卷积（每个通道独立卷积）
        - 再展开为序列形式 [B, N, C]
    """

    def __init__(self, hidden_features: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features, hidden_features,
                kernel_size=kernel_size, stride=1,
                padding=(kernel_size - 1) // 2, dilation=1,
                groups=hidden_features
            ),
            nn.GELU()
        )

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, H*W, C]
            x_size: 张量 x 的 H 和 W

        Returns:
            Tensor of shape [B, H*W, C]
        """
        H, W = x_size
        B, _, C = x.shape

        x = x.transpose(1, 2).view(B, C, H, W).contiguous()  # [B, H*W, C] ->[B, C, H*W] ->[B, C, H, W]
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, C, H, W] ->[B, C, H*W] ->[B, H*W, C]
        return x


class ConvFFN(nn.Module):
    """
    带有深度卷积增强的前馈网络模块（ConvFFN），常用于视觉任务中的Transformer结构。

    相较于标准的MLP结构，本模块通过引入深度可分离卷积（Depthwise Conv），融合了局部空间信息，提升模型对局部结构的建模能力。

    Args:
        in_features (int): 输入特征维度。
        hidden_features (int, 可选): 隐藏层特征维度，默认与输入维度一致。
        out_features (int, 可选): 输出特征维度，默认与输入维度一致。
        kernel_size (int, 可选): 深度卷积的卷积核大小，默认为 5。
        act_layer (nn.Module, 可选): 激活函数，默认为 GELU。
    """

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, kernel_size: int = 5,
                 act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dw_conv = DWConv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, H*W, C]
            x_size: 张量 x 的 H 和 W

        Returns:
            Tensor of shape [B, H*W, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dw_conv(x, x_size)
        x = self.fc2(x)
        return x


class BSConvU(torch.nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=1,
            dilation=1,
            bias=True,
            padding_mode="zeros",
            with_bn=False,
            bn_kwargs=None
    ):
        super().__init__()
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}
        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))
        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))
        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        ))


class MultiScaleMoEBlock(nn.Module):
    """
    MS-MoE:Multi-Scale Mixture-of-Experts Block
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.dim = dim
        hidden_dim = dim // reduction

        # 门控网络
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1),  # 输出3个专家权重
            nn.Softmax(dim=1)
        )

        # 专家1（V5 MoE使用）
        # self.exp1 = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        # )
        self.exp1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            BSConvU(hidden_dim, hidden_dim, 3, 1, 1),
            # nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
        )

        # 专家2
        self.exp2 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # DSA(hidden_dim, 5, 3, 2, 1, dilation=1, group=hidden_dim),
            BSConvU(hidden_dim, hidden_dim, 5, 2, 1),
            # BSConvU(hidden_dim, hidden_dim, 3, 1, 1),（V5 MoE使用）
            # nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
        )

        # 专家3
        self.exp3 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # DSA(hidden_dim, 7, 5, 3, 1, dilation=1, group=hidden_dim),
            BSConvU(hidden_dim, hidden_dim, 7, 3, 1),
            # BSConvU(hidden_dim, hidden_dim, 5, 2, 1),（V5 MoE使用）
            # nn.Conv2d(hidden_dim, hidden_dim, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )

        self.out_scale = nn.Parameter(torch.ones(1) * 0.001)

    def forward(self, x):
        # 门控权重计算
        weights = self.gate(x)  # [B,3,1,1]
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)

        # 专家输出
        out = w1 * self.exp1(x)
        out += w2 * self.exp2(x)
        out += w3 * self.exp3(x)

        return x + out * self.out_scale


class FourierDynamicWindowAttention(nn.Module):
    """
    FDWA：Fourier Dynamic Window Attention
    """

    def __init__(
            self,
            dim: int,
            layer_id,
            num_topk,
            window_size: int = 16,
            num_heads: int = 6,
            shift_size: int = 0,
            conv_ffn_kernel_size: int = 7,
            df_expansion_ratio: int = 1,
            df_num_filters: int = 2,
            compress_ratio: int = 3,
            squeeze_factor: int = 30,
            conv_scale: float = 0.01,
            mlp_ratio: float = 2.,
            qkv_bias: bool = True,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.v_LePE = DWConv(hidden_features=dim, kernel_size=conv_ffn_kernel_size)

        self.dynamic_filter = DynamicFilter(
            dim=dim,
            window_size=window_size,
            expansion_ratio=df_expansion_ratio,
            num_filters=df_num_filters,
        )

        self.conv_scale = conv_scale
        self.conv_block = ChannelAttentionBlock(
            num_feat=dim,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor
        )  # 通道注意力模块（CAB）

        self.attn_win = WindowAttention(
            dim=dim,
            layer_id=layer_id,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        self.conv_ffn = ConvFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            kernel_size=conv_ffn_kernel_size
        )

    def forward(
            self,
            x: Tensor,
            fba_list: list,
            x_size: tuple[int, int],
            params: dict[str, Tensor]
    ) -> tuple[Tensor, list]:
        """
        Args:
            x: Tensor of shape (B, H*W, C)
            fba_list:
            x_size: 张量 x 的 H 和 W
            params: SW-MSA参数 {'attn_mask':注意力掩码,'rpi_sa':相对位置索引}

        Returns:
            Tensor of shape (B, H*W, C)
        """
        fba_values, fba_indices = fba_list[0], fba_list[1]
        H, W = x_size
        B, N, C = x.shape
        C4 = 4 * C

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)  # [B, H*W, C]->[B, H, W, C]

        # # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2).contiguous())  # [B, H, W, C] ->[B, C, H, W]
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B, C, H, W] ->[B, H*W, C]

        # DynamicFilter
        # [B, H, W, C] ->[num_windows * B, window_size, window_size, C]
        x_windows = window_partition(x, self.window_size)
        x_windows = self.dynamic_filter(x_windows)
        # [num_windows * B, window_size, window_size, C] ->[B, H, W, C] ->[B, H*W, C]
        x = window_reverse(x_windows, self.window_size, H, W).view(B, H * W, C)

        # SW-MSA准备
        x_qkv = self.wqkv(x)
        v_lepe = self.v_LePE(torch.split(x_qkv, C, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        # SW-MSA
        # cyclic shift
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(B, H, W, C4),
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(B, H, W, C4)
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C4)  # nw*b, window_size*window_size, c
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, fba_values, fba_indices = self.attn_win(
            x_windows,
            pfa_values=fba_values,
            pfa_indices=fba_indices,
            rpi=params['rpi_sa'], mask=params['attn_mask'],
            shift=shift
        )
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # b h' w' c
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        # FFN
        x = shortcut + x_win.view(B, N, C) + conv_x * self.conv_scale
        x = x + self.conv_ffn(self.norm2(x), x_size)

        fba_list = [fba_values, fba_indices]
        return x, fba_list


class MoEPreAttention(nn.Module):
    """
    MEPA Block (Mixture-of-Experts Pre-Attention Block)
    """

    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = DWConv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.moe = MultiScaleMoEBlock(dim=dim)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x

        x = self.norm1(x)

        x = (self.moe(x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()).
             permute(0, 2, 3, 1).contiguous().view(b, h * w, c))

        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        # SW-MSA
        # cyclic shift
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, h, w, c4), shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)  # nw*b, window_size*window_size, c
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, pfa_values, pfa_indices = self.attn_win(x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices,
                                                              rpi=params['rpi_sa'], mask=params['attn_mask'],
                                                              shift=shift)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        x = shortcut + x_win.view(b, n, c)
        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list


class FMBlock(nn.Module):
    def __init__(
            self,
            block_id: int,
            layer_id: int,
            num_topk: tuple[int, ...],
            dim: int,
            input_resolution: tuple[int, int] = (64, 64),
            window_size: int = 16,
            num_heads: int = 6,
            mlp_ratio: float = 2.,
            conv_ffn_kernel_size: int = 5,
            df_expansion_ratio: int = 1,
            df_num_filters: int = 2,
            compress_ratio: int = 3,
            squeeze_factor: int = 30,
            qkv_bias: bool = False,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.fdwa_block = FourierDynamicWindowAttention(
            dim=dim,
            layer_id=layer_id,
            num_topk=num_topk,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=0,
            conv_ffn_kernel_size=conv_ffn_kernel_size,
            df_expansion_ratio=df_expansion_ratio,
            df_num_filters=df_num_filters,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        self.mepa_block = MoEPreAttention(
            dim=dim,
            block_id=block_id,
            layer_id=layer_id,
            input_resolution=input_resolution,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            shift_size=window_size // 2,
            convffn_kernel_size=conv_ffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(
            self,
            x: Tensor,
            x_size: tuple[int, int],
            params: dict[str, Tensor],
            pfa_list: list,
            fba_list: list,
    ) -> tuple[Tensor, list, list]:
        """
        Args:
            x: Tensor of shape (B, H*W, C)
            x_size: 张量 x 的 H 和 W
            params: SW-MSA参数 {'attn_mask':注意力掩码,'rpi_sa':相对位置索引}
            pfa_list: PFT共享Attention参数
            fba_list: FB共享Attention参数

        Returns:
            Tensor of shape (B, H*W, C)
        """
        # Part 1: FourierWindowAttnBlock
        res = x
        x, fba_list = self.fdwa_block(x, fba_list, x_size, params)
        x = x + (res * self.scale1)

        # Part 2: PFTransformerLayer
        res = x
        x, pfa_list = self.mepa_block(x, pfa_list, x_size, params)
        x = x + (res * self.scale2)

        return x, pfa_list, fba_list


class EFMGroup(nn.Module):
    """
    EFM-RG:Enhanced Fourier-Mixture Residual Group
    """

    def __init__(
            self,
            block_id: int,
            layer_id: int,
            dept: int,
            num_topk: tuple[int, ...],
            dim: int,
            input_resolution: tuple[int, int] = (64, 64),
            window_size: int = 16,
            num_heads: int = 6,
            mlp_ratio: float = 2.,
            conv_ffn_kernel_size: int = 5,
            df_expansion_ratio: int = 1,
            df_num_filters: int = 2,
            compress_ratio: int = 3,
            squeeze_factor: int = 30,
            qkv_bias: bool = False,
            norm_layer=nn.LayerNorm,
            img_size: int = 64,
            patch_size: int = 1,
            resi_connection='1conv'
    ):
        super().__init__()

        self.resi_connection = resi_connection

        if self.resi_connection not in ['1conv', '3conv']:
            raise NotImplementedError

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim
        )

        self.blocks = nn.ModuleList()
        for i in range(dept):
            self.blocks.append(
                FMBlock(
                    block_id=block_id,
                    layer_id=layer_id + i,
                    num_topk=num_topk,
                    dim=dim,
                    input_resolution=input_resolution,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    conv_ffn_kernel_size=conv_ffn_kernel_size,
                    df_expansion_ratio=df_expansion_ratio,
                    df_num_filters=df_num_filters,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        if self.resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif self.resi_connection == '3conv':
            # 为了减少参数和内存
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

    def forward(
            self,
            x: Tensor,
            x_size: tuple[int, int],
            params: dict[str, Tensor],
            pfa_list: list,
            fba_list: list
    ) -> tuple[Tensor, list, list]:
        """
        Args:
            x: Tensor of shape (B, H*W, C)
            x_size: 张量 x 的 H 和 W
            params: SW-MSA参数 {'attn_mask':注意力掩码,'rpi_sa':相对位置索引}
            pfa_list: 共享Attention参数
            fba_list: FB共享Attention参数

        Returns:
            Tensor of shape (B, H*W, C)
        """
        res = x
        for block in self.blocks:
            x, pfa_list, fba_list = block(x, x_size, params, pfa_list, fba_list)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        x = x + res
        return x, pfa_list, fba_list


@ARCH_REGISTRY.register()
class EFMSR(nn.Module):
    def __init__(
            self,
            upscale: int = 2,  # 放大倍率
            img_size: int = 64,  # 输入图像大小（高宽相等）
            patch_size: int = 1,  # Patch 划分大小（1 表示不划分）
            img_range: float = 1.,
            in_chans: int = 3,  # 输入通道
            embed_dim: int = 180,  # 嵌入维度，用于主干网络中的特征表示
            depths: tuple[int, ...] = (2, 2, 2, 3, 3, 3),  # 长度代表Block个数，内容代表一个Block内的内容重复次数
            resi_connection: str = '1conv',  # Blocks后卷积层类型，支持 '1conv'、'3conv'（轻量化）
            num_feat: int = 64,  # 特征通道数，用于上采样和重建阶段
            upsampler: str = 'pixel_shuffle',  # 上采样方式，支持 'pixel_shuffle'、'pixel_shuffle_direct'（轻量化）
            norm_layer=nn.LayerNorm,
            ape=False,
            window_size: int = 16,
            num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
            num_topk: tuple[int, ...] = (
                    1024, 1024,
                    512, 512,
                    256, 256,
                    128, 128, 128,
                    64, 64, 64,
                    32, 32, 32,
            ),
            mlp_ratio=2.,  # MLP 扩展比例（Transformer 中常用）
            conv_ffn_kernel_size: int = 7,
            df_expansion_ratio: int = 1,
            df_num_filters: int = 2,
            compress_ratio: int = 12,
            squeeze_factor: int = 30,
            qkv_bias=True,  # QKV 的 Linear 是否使用偏置
    ):
        super().__init__()
        self.upscale = upscale
        self.num_in_ch = in_chans
        self.num_out_ch = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.resi_connection = resi_connection
        self.num_feat = num_feat
        self.img_range = img_range
        self.upsampler = upsampler

        if self.num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        if self.upsampler not in ["pixel_shuffle", "pixel_shuffle_direct"]:
            raise NotImplementedError

        if self.resi_connection not in ["1conv", "3conv"]:
            raise NotImplementedError

        # ------------------------- 1, 浅层特征提取 ------------------------- #
        self.conv_first = nn.Conv2d(self.num_in_ch, self.embed_dim, 3, 1, 1)

        # ------------------------- 2, 深层特征提取 ------------------------- #
        self.num_layers = len(depths)
        self.layer_id = 0
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # 图像转Embedding [B, C, H, W] → [B, H*W, C]
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 图像到Patch解嵌入 [B, H*W, C] → [B, C, H, W]
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, int(num_patches), embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # SW-WSA相对位置索引
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layer_id = 0
        self.groups = nn.ModuleList()
        for i in range(len(self.depths)):
            self.groups.append(
                EFMGroup(
                    block_id=i,
                    layer_id=self.layer_id,
                    dept=depths[i],
                    num_topk=num_topk,
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    window_size=window_size,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    conv_ffn_kernel_size=conv_ffn_kernel_size,
                    df_expansion_ratio=df_expansion_ratio,
                    df_num_filters=df_num_filters,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    qkv_bias=qkv_bias,
                    norm_layer=nn.LayerNorm,
                )
            )
            self.layer_id = self.layer_id + depths[i]

        self.norm = norm_layer(self.embed_dim)

        # 构建深层特征提取模块中的最后一个卷积层
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 为了减少参数和内存
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        # -------------------------3. 高质量图像重建 ------------------------ #
        if self.upsampler == 'pixel_shuffle':
            # 用于经典SR
            self.reconstruction = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.num_feat, 3, 1, 1),
                    nn.LeakyReLU(inplace=True)
                ),  # conv_before_upsample
                Upsample(self.upscale, self.num_feat),  # upsample
                nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1),  # conv_last
            )
        elif self.upsampler == 'pixel_shuffle_direct':
            # 用于轻量化SR（节省参数）
            self.reconstruction = nn.Sequential(
                UpsampleOneStep(self.upscale, self.embed_dim, self.num_out_ch),  # conv_last
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 需要完全排除 weight_decay（权重衰减）的参数名。
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    # 需要模糊排除 weight_decay（权重衰减）的参数名关键字。
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def calculate_rpi_sa(self):
        # 获取窗口中每个 token 对之间的相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # 计算用于 Shifted Window Multi-Head Self-Attention 的注意力掩码
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward_features(self, x: Tensor, params: dict) -> Tensor:
        """Block块循环处理

        Args:
            x: Tensor of shape (B, C, H, W)
            params: SW-MSA参数 {'attn_mask':注意力掩码,'rpi_sa':相对位置索引}

        Returns:
            Tensor of shape (B, C, H, W)

        """
        x_size = (x.shape[2], x.shape[3])  # 保存原图像[H, W]

        # Define progressive focusing attention (PFA) values and their corresponding indices
        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        # 运用到FB上
        fba_values = [None, None]
        fba_indices = [None, None]
        fba_list = [fba_values, fba_indices]

        x = self.patch_embed(x)  # [B, C, H, W] → [B, H*W, C]
        for group in self.groups:
            x, pfa_list, fba_list = group(x, x_size, params, pfa_list, fba_list)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)  # [B, H*W, C] → [B, C, H, W]
        return x

    def forward(self, x: Tensor) -> Tensor:
        # 保证输入高宽是窗口大小的整数倍，做镜像填充
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        # 对输入做均值归一化处理
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        skip = x
        # ------------------------- 1, 浅层特征提取 ------------------------- #
        x = self.conv_first(x)
        # ------------------------- 2, 深层特征提取 ------------------------- #
        x = self.conv_after_body(self.forward_features(x, params)) + x
        # -------------------------3. 高质量图像重建 ------------------------ #
        x = x + torch.repeat_interleave(skip, repeats=self.embed_dim // 3, dim=1)
        x = self.reconstruction(x)

        x = x / self.img_range + self.mean  # 反归一化
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]  # 去除 padding

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64).cuda()
    # model = EFMSR().cuda()
    # print(model)

    model = EFMSR(
        embed_dim=48,
        depths=(1, 2, 3, 3, 3),
        upsampler='pixel_shuffle_direct',
        num_heads=(4, 4, 4, 4, 4),
        num_topk=(
            1024,
            256, 256,
            128, 128, 128,
            64, 64, 64,
            32, 32, 32,
        )
    ).cuda()
    print(model)

    flops, params = profile(model=model, inputs=(x,))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"Number of parameter: {params}")
    print(f"Number of flops: {flops}")
    print(parameter_count_table(model, max_depth=6))
