import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            # print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

            x = self.norm(x)
            x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv = nn.Conv2d(in_channels=4 * self.d_inner, out_channels=self.d_inner, kernel_size=1)
        self.con = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.LN = nn.LayerNorm(self.d_model)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.contiguous().view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).contiguous().view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # print('xs', xs.view(B, K, -1, L).shape)  # 打印第一个操作数的形状
        # print('weight', self.x_proj_weight.shape)  # 打印第二个操作数的形状
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


import torch
import math
from functools import partial
from typing import Callable, Optional
import copy
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint


class FrequencyAttention(nn.Module):
    """频域注意力机制（2025年ISPRS顶刊提出）"""

    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, dim // reduction_ratio, bias=False)
        self.expansion = nn.Linear(dim // reduction_ratio, dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 保存原始输入用于残差连接
        identity = x

        # 转换到频域
        x_fft = fft.rfft2(x, dim=(1, 2), norm='ortho')
        magnitude = torch.abs(x_fft)

        # 全局平均池化
        gap = magnitude.mean(dim=(1, 2))

        # 频域特征压缩与扩展
        weights = self.reduction(gap)
        weights = F.relu(weights)
        weights = self.expansion(weights)
        weights = self.sigmoid(weights)

        # 应用注意力权重
        weights = weights.view(-1, 1, 1, self.dim)
        return identity * weights


class CrossLayerCrossAttention(nn.Module):
    """跨层交叉注意力机制（IEEE TGRS 2024）"""

    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, H, W, C = x.shape
        q = self.q(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 处理上下文特征
        context = F.interpolate(context.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ====================== 新增改进点：显著特征抑制模块 ======================
class SaliencyFeatureSuppression(nn.Module):
    """显著特征抑制模块 (Nature Scientific Reports 2024) [1](@ref)"""

    def __init__(self, suppression_ratio=0.2, suppression_radius=1):
        super().__init__()
        self.suppression_ratio = suppression_ratio
        self.suppression_radius = suppression_radius

    def forward(self, x):
        # 保留原始特征用于残差连接
        identity = x

        # 计算特征图的显著性 (绝对值)
        saliency = torch.abs(x)

        # 计算每个空间位置的显著性得分 (通道维度平均)
        spatial_saliency = torch.mean(saliency, dim=-1, keepdim=True)  # [B, H, W, 1]

        # 获取显著区域位置 (top k% 最显著的位置)
        B, H, W, _ = spatial_saliency.shape
        k = int(H * W * self.suppression_ratio)
        topk_values, topk_indices = torch.topk(spatial_saliency.view(B, -1), k, dim=-1)

        # 创建抑制掩码
        mask = torch.ones_like(spatial_saliency)
        for b in range(B):
            # 将一维索引转换为二维坐标
            indices = topk_indices[b]
            rows = indices // W
            cols = indices % W

            # 创建局部抑制区域
            for r in range(-self.suppression_radius, self.suppression_radius + 1):
                for c in range(-self.suppression_radius, self.suppression_radius + 1):
                    new_rows = torch.clamp(rows + r, 0, H - 1)
                    new_cols = torch.clamp(cols + c, 0, W - 1)

                    # 更新掩码
                    mask[b, new_rows, new_cols, :] = 0.1

        # 应用抑制 (保留10%的特征值)
        x = x * mask

        return identity + (x - identity).detach()


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            use_frequency_attention: bool = True,  # 频域注意力开关
            use_cross_layer_attention: bool = False,  # 跨层注意力开关
            use_saliency_suppression: bool = True,  # 新增显著特征抑制开关
            **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim  # 保存通道数用于验证
        self.ln_1 = norm_layer(hidden_dim)

        # 频域注意力模块
        self.use_frequency_attention = use_frequency_attention
        if self.use_frequency_attention:
            self.freq_attn = FrequencyAttention(hidden_dim)

        # 跨层交叉注意力模块
        self.use_cross_layer_attention = use_cross_layer_attention
        if self.use_cross_layer_attention:
            self.cross_attn = CrossLayerCrossAttention(
                dim=hidden_dim,
                num_heads=4,
                attn_drop=attn_drop_rate,
                proj_drop=attn_drop_rate
            )

        # 显著特征抑制模块
        self.use_saliency_suppression = use_saliency_suppression
        if self.use_saliency_suppression:
            self.saliency_suppression = SaliencyFeatureSuppression(suppression_ratio=0.2)

        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 新增残差连接后的归一化层
        self.ln_2 = norm_layer(hidden_dim) if use_cross_layer_attention else nn.Identity()

    def forward(self, input: torch.Tensor, context: Optional[torch.Tensor] = None):
        # 添加残差连接
        x = input

        # 应用LayerNorm
        x_norm = self.ln_1(x)

        # 应用频域注意力
        if self.use_frequency_attention:
            x_norm = self.freq_attn(x_norm)

        # 空间状态扫描
        x_attn = self.self_attention(x_norm)

        # 应用显著特征抑制
        if self.use_saliency_suppression:
            x_attn = self.saliency_suppression(x_attn)

        # 残差连接
        x = input + self.drop_path(x_attn)

        # 应用跨层交叉注意力
        if self.use_cross_layer_attention and context is not None:
            x = self.ln_2(x)
            x_cross = self.cross_attn(x, context)
            x = x + x_cross

        return x


class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            use_frequency_attention=True,  # 频域注意力开关
            use_cross_layer_attention=False,  # 跨层注意力开关
            use_saliency_suppression=True,  # 新增显著特征抑制开关
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.use_cross_layer_attention = use_cross_layer_attention

        # 存储前一层的特征（用于跨层注意力）
        self.prev_features = None

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_frequency_attention=use_frequency_attention,
                use_cross_layer_attention=use_cross_layer_attention and (i == depth - 1),  # 仅在最后一个块使用
                use_saliency_suppression=use_saliency_suppression,  # 传递参数
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, prev_features=None):
        # 保存当前层的特征（用于下一层的跨层注意力）
        if self.use_cross_layer_attention:
            self.prev_features = x.detach().clone() if prev_features is None else prev_features

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                # 为最后一个块提供上下文信息
                if i == len(self.blocks) - 1 and self.use_cross_layer_attention:
                    x = checkpoint.checkpoint(blk, x, self.prev_features)
                else:
                    x = checkpoint.checkpoint(blk, x)
            else:
                # 为最后一个块提供上下文信息
                if i == len(self.blocks) - 1 and self.use_cross_layer_attention:
                    x = blk(x, self.prev_features)
                else:
                    x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSM(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 depths: list = [2, 2, 2, 2],
                 dims: list = [96, 96, 96, 96],
                 d_state: int = 16,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer: Callable = nn.LayerNorm,
                 patch_norm: bool = True,
                 use_checkpoint: bool = False,
                 use_frequency_attention: bool = True,  # 频域注意力开关
                 use_cross_layer_attention: bool = True,  # 跨层注意力开关
                 use_saliency_suppression: bool = True,  # 新增显著特征抑制开关
                 **kwargs):
        super().__init__()

        # 基础参数设置
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dims = dims
        self.num_features = dims[-1]
        self.use_cross_layer_attention = use_cross_layer_attention

        # 图像分块嵌入层
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None
        )

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 存储各层特征（用于跨层注意力）
        self.layer_features = [None] * self.num_layers

        # 构建网络层次
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_frequency_attention=use_frequency_attention,
                use_cross_layer_attention=use_cross_layer_attention and (i_layer > 0),  # 从第二层开始
                use_saliency_suppression=use_saliency_suppression,  # 传递参数
            )
            self.layers.append(layer)

        # 分类头
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # 重置各层特征存储
        if self.use_cross_layer_attention:
            self.layer_features = [None] * self.num_layers

        # 存储第一层特征
        if self.use_cross_layer_attention:
            self.layer_features[0] = x.detach().clone()

        for i, layer in enumerate(self.layers):
            # 为当前层提供前一层的特征
            prev_features = self.layer_features[i - 1] if i > 0 else None
            x = layer(x, prev_features)

            # 存储当前层特征
            if self.use_cross_layer_attention and i < self.num_layers - 1:
                self.layer_features[i] = x.detach().clone()

        x = self.norm(x)  # 最终归一化 [B, H, W, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # 全局平均池化 [B, H, W, C] -> [B, C]
        x = x.mean(dim=(1, 2))
        # 分类头
        x = self.head(x)
        return x


if __name__ == "__main__":
    # check_vssm_equals_vmambadp()
    model = VSSM().to('cuda')
    int = torch.randn(16, 3, 224, 224).cuda()  # 标准输入尺寸
    out = model(int)
    print(out.shape)