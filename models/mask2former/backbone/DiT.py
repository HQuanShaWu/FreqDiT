import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from functools import partial
from collections import namedtuple
from tqdm import tqdm
from einops import rearrange
from random import random
from types import SimpleNamespace

from torchmetrics.functional import structural_similarity_index_measure as ssim
from fusion.utils import fit_utils
from fusion.utils.losses import fusion_loss_vif
from timm.models.layers.helpers import to_2tuple
import numpy as np
from dataloaders.FMB_dataset import FMB_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from functools import partial


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, H, W)
    grid = np.stack(grid, axis=0)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    def get_1d(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = get_1d(embed_dim // 2, grid[0])
    emb_w = get_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SobelMultiScale(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super().__init__()
        kernels = []
        base = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32)
        for k in kernel_sizes:
            kernel = F.interpolate(
                base.unsqueeze(0).unsqueeze(0),
                size=(k, k), mode='bilinear', align_corners=False
            )
            kernel = kernel / (kernel.abs().sum() + 1e-8)
            kernels.append(nn.Parameter(kernel, requires_grad=False))
        self.kernels = nn.ParameterList(kernels)

    def forward(self, x):
        grads = []
        for k in self.kernels:
            kx = k.to(x.device)
            ky = kx.flip(2).transpose(2, 3)
            kx = kx.repeat(x.shape[1], 1, 1, 1)
            ky = ky.repeat(x.shape[1], 1, 1, 1)
            grad_x = F.conv2d(x, kx, padding=kx.shape[-1] // 2, groups=x.shape[1])
            grad_y = F.conv2d(x, ky, padding=ky.shape[-1] // 2, groups=x.shape[1])
            grads.append(torch.abs(grad_x) + torch.abs(grad_y))
        return grads


class FusionLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_grad=1.0, lambda_ssim=1.0):
        super(FusionLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.grad_extractor = SobelMultiScale([3,5,7])

    def forward(self, fused, x, y, m1=None, m2=None):
        if m1 is None: m1 = torch.ones_like(x)
        if m2 is None: m2 = torch.ones_like(y)

        if fused.shape[1] != x.shape[1]:
            x = x.repeat(1, fused.shape[1], 1, 1)
        if fused.shape[1] != y.shape[1]:
            y = y.repeat(1, fused.shape[1], 1, 1)

        loss_mse = F.mse_loss(fused, m1 * x) + F.mse_loss(fused, m2 * y)

        grads_f = self.grad_extractor(fused)
        grads_x = self.grad_extractor(x)
        grads_y = self.grad_extractor(y)
        loss_grad = 0
        for g_f, g_x, g_y in zip(grads_f, grads_x, grads_y):
            loss_grad += F.mse_loss(g_f, torch.max(g_x, g_y))

        loss_ssim = 0.5 * ((1 - ssim(fused, x)) + (1 - ssim(fused, y)))

        total_loss = (
            self.lambda_mse * loss_mse +
            self.lambda_grad * loss_grad +
            self.lambda_ssim * loss_ssim
        )
        return total_loss
    
class TimestepFrequencyFilter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.t2p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()
        )
        self.t2cut = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, 2),
            nn.Sigmoid()
        )

    def forward(self, cond, t_emb, h=None):
        B, C, Hc, Wc = cond.shape
        device = cond.device

        p = self.t2p(t_emb)
        c = self.t2cut(t_emb)
        c_lp, c_hp = c[:, :1], c[:, 1:]

        rr = self._make_radial_grid(Hc, Wc, device)
        M_lp = self._butterworth(rr, cutoff=c_lp.view(-1,1,1), order=4, mode='low')
        M_hp = self._butterworth(rr, cutoff=c_hp.view(-1,1,1), order=4, mode='high')
        M_lp = M_lp.view(B,1,Hc,Wc)
        M_hp = M_hp.view(B,1,Hc,Wc)

        cond_low  = self._fft_filter(cond, M_lp)
        cond_high = self._fft_filter(cond, M_hp)

        p_bhw = p.view(B,1,1,1)
        cond_f = p_bhw * cond_low + (1 - p_bhw) * cond_high

        if h is not None:
            cond_f = F.interpolate(cond_f, size=h.shape[-2:], mode='bilinear', align_corners=False)

        return cond_f

    def _make_radial_grid(self, H, W, device):
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        rr = torch.sqrt((xx - W//2)**2 + (yy - H//2)**2)
        rr = rr / rr.max()
        return rr

    def _butterworth(self, rr, cutoff, order=4, mode='low'):
        if mode == 'low':
            return 1 / (1 + (rr / cutoff)**(2 * order))
        else:
            return 1 - 1 / (1 + (rr / cutoff)**(2 * order))

    def _fft_filter(self, x, mask):
        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
        Xf = Xf * mask
        x_filt = torch.fft.ifft2(torch.fft.ifftshift(Xf), norm="ortho").real
        return x_filt
    
    def _init_weights(self):
        for layer in self.t2p:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.t2p[-2].weight.data += torch.abs(
            torch.randn_like(self.t2p[-2].weight)
        ) * 0.1
        if self.t2p[-2].bias is not None:
            self.t2p[-2].bias.data.fill_(0.5)

        for layer in self.t2cut:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.t2cut[-2].weight.data -= torch.abs(
            torch.randn_like(self.t2cut[-2].weight)
        ) * 0.1
        if self.t2cut[-2].bias is not None:
            self.t2cut[-2].bias.data.fill_(0.5)
    

def init():
    class parser:
        epochs = 1000
        G_optimizer_lr = 2e-5
        batch_size = 8
        device = 'cuda'
        is_train = True
        E_decay = .999
        G_regularizer_orthstep = None
        G_regularizer_clipstep = None
        G_optimizer_clipgrad = None
        G_scheduler_milestones = [
            250000,
            400000,
            450000,
            475000,
            500000
        ]
        G_scheduler_gamma = .5
        G_lossfn_weight = 1.
        G_optimizer_reuse = True

        save_dir = './checkpoint_mat_rgb'
        checkpoint_save = 1

    return parser()



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq = freq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(float(max_period))) *\
                        torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq))
    

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        s1, g1, m1, s2, g2, m2 = self.ada(c).chunk(6, dim=1)  # shift/scale/gate for attn/mlp
        x = x + m1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, g1),
                                            modulate(self.norm1(x), s1, g1),
                                            modulate(self.norm1(x), s1, g1),
                                            need_weights=False)[0]
        x = x + m2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, g2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        s, g = self.ada(c).chunk(2, dim=1)
        x = modulate(self.norm(x), s, g)
        return self.head(x)
    

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=None,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MultiScaleFusionHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, 
                          padding=s, dilation=s),
                nn.SiLU(),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, 
                          padding=s, dilation=s),
                nn.SiLU()
            ) for s in scales
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels // 4 * len(scales), in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, len(scales)),
            nn.Softmax(dim=1)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 8, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        multi_feats = [branch(x) for branch in self.branches]
        stacked = torch.stack(multi_feats, dim=1)
        pooled = torch.cat([self.global_pool(f).flatten(1) for f in multi_feats], dim=1)
        weights = self.channel_attn(pooled).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = (stacked * weights).sum(1)
        return self.fuse_conv(fused)


class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_ir = nn.LayerNorm(dim)
        self.cross_attn_rgb = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_ir = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, rgb_tok, ir_tok):
        rgb_norm = self.norm_rgb(rgb_tok)
        ir_norm = self.norm_ir(ir_tok)

        rgb_fused, _ = self.cross_attn_rgb(rgb_norm, ir_norm, ir_norm)
        ir_fused, _ = self.cross_attn_ir(ir_norm, rgb_norm, rgb_norm)

        rgb_out = rgb_tok + self.drop(rgb_fused)
        ir_out = ir_tok + self.drop(ir_fused)

        rgb_out = rgb_out + self.ffn(rgb_out)
        ir_out = ir_out + self.ffn(ir_out)

        return rgb_out, ir_out
    

class DiT(nn.Module):
    def __init__(
        self,
        patch_size=8,
        rgb_channels=3,
        ir_channels=1,
        cond_emb_channels=1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        cross_layers=4,
    ):
        super().__init__()
        self.patch = patch_size
        self.hidden = hidden_size
        self.cross_layers = cross_layers
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.cond_emb_channels = cond_emb_channels
        self.rgb_embedder = PatchEmbed(None, patch_size, rgb_channels, hidden_size)
        self.ir_embedder  = PatchEmbed(None, patch_size, ir_channels, hidden_size)
        self.emb_embedder = PatchEmbed(None, patch_size, cond_emb_channels, hidden_size)
        self.t_embedder   = TimestepEmbedder(hidden_size)
        self.cond_filter = TimestepFrequencyFilter(hidden_dim=self.hidden)

        self.cross_blocks = nn.ModuleList([
            CrossModalBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(cross_layers)
        ])
        self.main_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth - cross_layers)
        ])

        self.noise_head = FinalLayer(hidden_size, patch_size, out_ch=rgb_channels + ir_channels)

        self.args = init()
        self.model_plain = fit_utils.ModelPlain(self.args)
        self.model_plain.init_train()
        self.fusion_head = self.model_plain.netG

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self, "cond_filter"):
            self.cond_filter._init_weights()

    def unpatchify(self, tokens, H, W, C):
        B, T, D = tokens.shape
        p = self.patch
        h, w = H // p, W // p
        assert T == h * w, f"Token count mismatch: T={T}, h*w={h*w}"
        assert D == p * p * C, f"Channel mismatch: D={D}, p^2*C={p*p*C}"

        x = tokens.view(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, C, H, W) 

    def forward(self, rgb, ir, t, cond=None):
        B, _, H, W = rgb.shape
        rgb_tok = self.rgb_embedder(rgb)
        ir_tok  = self.ir_embedder(ir)
        rgb_patch_num = rgb_tok.shape[1]

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=rgb.device).repeat(B)
        t_emb = self.t_embedder(t.long())

        if cond is not None:
            cond_filtered = self.cond_filter(cond, t_emb)
            cond_tok = self.emb_embedder(cond_filtered)
            rgb_tok = torch.cat([rgb_tok, cond_tok], dim=1)
            ir_tok  = torch.cat([ir_tok, cond_tok], dim=1)

        pos = torch.from_numpy(get_2d_sincos_pos_embed(self.hidden, (H // self.patch, W // self.patch))).float().to(rgb.device)
        pos = pos.unsqueeze(0)
        rgb_tok[:, :rgb_patch_num] += pos
        ir_tok[:, :rgb_patch_num]  += pos

        for blk in self.cross_blocks:
            rgb_tok, ir_tok = blk(rgb_tok, ir_tok)

        tokens = torch.cat([rgb_tok, ir_tok], dim=1)

        h = tokens
        for blk in self.main_blocks:
            h = blk(h, t_emb)

        out_tokens = self.noise_head(h[:, :rgb_patch_num], t_emb)
        eps_hat = self.unpatchify(out_tokens, H, W, self.rgb_channels + self.ir_channels)

        feat_map = h[:, :rgb_patch_num]
        ph, pw = H // self.patch, W // self.patch
        feat_map = feat_map.transpose(1, 2).reshape(B, self.hidden, ph, pw)
        feat_map = F.interpolate(feat_map, size=(H, W), mode="bilinear", align_corners=False)
        fused = self.fusion_head(feat_map, cond[:,:3], cond[:,-1:].repeat(1, 3, 1, 1))

        return eps_hat, fused



# 假设以下类你已经定义好了 (从你的代码中继承)
# TimestepEmbedder, TimestepFrequencyFilter, init, fit_utils

class ResBlock(nn.Module):
    """标准的 ResNet Block，支持 Time Embedding 注入"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 时间步投影层，将 time_emb 映射到 feature map 的通道数
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        
        # 注入时间步信息 (Scale & Shift 也可以，这里用简单的加法)
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t_emb=None):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """用于瓶颈层的 Self-Attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, t_emb=None):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).permute(0, 2, 1) # B, HW, C
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h

class FusionUNet(nn.Module):
    def __init__(
        self,
        rgb_channels=3,
        ir_channels=1,
        cond_emb_channels=1, # 注意：UNet通常直接cat条件图，这里对应cond的通道数
        base_channels=64,    # UNet的基础通道数，对应DiT的hidden_size概念
        channel_mults=(1, 2, 4, 4), # 通道倍率
        time_emb_dim=256,    # 时间步嵌入维度
    ):
        super().__init__()
        
        # 1. 参数保存
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.cond_emb_channels = cond_emb_channels
        self.hidden = base_channels * channel_mults[-1] # 用于对齐 fusion_head 的输入维度

        # 2. 辅助模块 (保持与 DiT 一致)
        self.t_embedder = TimestepEmbedder(time_emb_dim)
        self.cond_filter = TimestepFrequencyFilter(hidden_dim=time_emb_dim) 
        # 注意：DiT中filter hidden是hidden_size，这里调整为time_emb_dim以匹配t_emb的输入

        # 3. UNet 结构构建
        # 输入通道 = RGB + IR + Cond (Filtered)
        input_channels = rgb_channels + ir_channels + cond_emb_channels
        
        self.inc = nn.Conv2d(input_channels, base_channels, 3, padding=1)

        # Downsampling Path
        self.downs = nn.ModuleList()
        cur_ch = base_channels
        feat_channels = [cur_ch] # 记录Skip connection通道

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            # 每个level放两个ResBlock
            for _ in range(2):
                self.downs.append(ResBlock(cur_ch, out_ch, time_emb_dim))
                cur_ch = out_ch
                feat_channels.append(cur_ch)
            
            # 如果不是最后一层，添加下采样
            if level != len(channel_mults) - 1:
                self.downs.append(Downsample(cur_ch, cur_ch))
                feat_channels.append(cur_ch)

        # Middle Path (Bottleneck)
        self.mid_block1 = ResBlock(cur_ch, cur_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(cur_ch)
        self.mid_block2 = ResBlock(cur_ch, cur_ch, time_emb_dim)

        # Upsampling Path
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            # 如果不是最后一层（倒序的第一层），添加上采样
            if level != len(channel_mults) - 1:
                self.ups.append(Upsample(cur_ch, cur_ch))
            
            # 每个level对应的ResBlock (处理 Skip Connection)
            for _ in range(2):
                # Skip connection 来自 down path，通道数需要相加
                skip_ch = feat_channels.pop()
                self.ups.append(ResBlock(cur_ch + skip_ch, out_ch, time_emb_dim))
                cur_ch = out_ch

        # 4. 输出头
        self.norm_out = nn.GroupNorm(32, cur_ch)
        self.act_out = nn.SiLU()
        
        # 预测噪声的头
        self.noise_head = nn.Conv2d(cur_ch, rgb_channels + ir_channels, 3, padding=1)

        # 5. Fusion Head (保持原样)
        self.args = init()
        self.model_plain = fit_utils.ModelPlain(self.args)
        self.model_plain.init_train()
        self.fusion_head = self.model_plain.netG
        
        # 特征投影层：UNet最后的特征维度可能和fusion_head需要的不一致
        # 假设 fusion_head 需要的维度是 128 (根据你的DiT配置)，而UNet输出是64
        # 我们需要从 bottleneck 或者最后的层提取特征。
        # 为了效果最好，通常取 decoder 最后一层的特征，但维度要匹配。
        fusion_in_dim = cur_ch # UNet输出层的通道数
        fusion_target_dim = 128 # 假设 fit_utils 里面定义的大小，如果不同请修改
        
        self.feature_proj = nn.Conv2d(fusion_in_dim, fusion_target_dim, 1) if fusion_in_dim != fusion_target_dim else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self, "cond_filter"):
            self.cond_filter._init_weights()

    def forward(self, rgb, ir, t, cond=None):
        B, _, H, W = rgb.shape
        
        # 1. 时间步嵌入
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=rgb.device).repeat(B)
        t_emb = self.t_embedder(t.long()) # (B, time_emb_dim)

        # 2. 条件处理
        cond_feat = cond
        if cond is not None:
            # 使用原有的滤波器处理条件
            # 注意：这里假设 cond_filter 输出保持空间维度 (B, C, H, W)
            cond_feat = self.cond_filter(cond, t_emb, h=rgb) 
        
        # 3. 拼接输入 (UNet 典型做法：Input Concatenation)
        # 将 RGB, IR 和 处理后的 Condition 在通道维度拼接
        x = torch.cat([rgb, ir, cond_feat], dim=1)

        # 4. UNet Forward
        skips = []
        h = self.inc(x)
        
        # Down
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h) # 2个ResBlock后不一定存skip，具体看结构，这里简单起见全存，Up的时候对应取
            elif isinstance(layer, Downsample):
                skips.append(h) # Downsample前的特征存一下
                h = layer(h)
            else:
                h = layer(h)

        # Mid
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up
        # 这里的逻辑需要小心匹配 skips 列表
        for layer in self.ups:
            if isinstance(layer, Upsample):
                h = layer(h)
            elif isinstance(layer, ResBlock):
                skip = skips.pop()
                # 如果由于padding导致尺寸不一致，进行裁剪或插值
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)

        # 5. Output Heads
        h = self.act_out(self.norm_out(h))
        
        # 输出噪声预测
        eps_hat = self.noise_head(h)

        # 输出融合图像
        # 提取特征图给 fusion_head
        feat_map = self.feature_proj(h) 
        # 注意：DiT 代码中 feat_map 经过了 interpolate，这里 UNet 输出已经是原图尺寸，直接用即可
        # 如果 fusion_head 需要特定尺寸，可在此 interpolate
        
        fused = self.fusion_head(feat_map, cond[:,:3], cond[:,-1:].repeat(1, 3, 1, 1))

        return eps_hat, fused


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SegDiff(Module):
    def __init__(
            self,
            model,
            *,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            input_img_channels=4,
            ddim_sampling_eta=1.,
    ):
        super().__init__()
        self.model = model
        self.input_img_channels = input_img_channels
        self.self_condition = False
        self.fusion_lossfn = FusionLoss()
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'},\
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v \
            (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        model_output, fused_img = self.model(x[:,:3,...], x[:,-1:,...], t, c)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), fused_img

    def p_mean_variance(self, x, t, c, x_self_cond=None, clip_denoised=False):
        preds, fused_img = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, fused_img

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, clip_denoised=False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, fused_img = self.p_mean_variance(x=x, t=batched_times, c=c,
                                                                                x_self_cond=x_self_cond,
                                                                                clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, fused_img

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, fused_img = self.p_sample(img, t, cond, self_cond)

        # img = unnormalize_to_zero_to_one(img)
        return img, fused_img

    @torch.no_grad()
    def ddim_sample(self, shape, cond, clip_denoised=False):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        # for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds, fused_img = self.model_predictions(img, time_cond, cond, self_cond,
                                                    clip_x_start=clip_denoised)

            x_start = preds.pred_x_start
            pred_noise = preds.pred_noise

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img, fused_img

    @torch.no_grad()
    def sample(self, cond):
        batch_size, device = cond.shape[0], self.device
        H, W = cond.shape[2], cond.shape[3]
        cond = cond.to(self.device)
        mask_channels = self.model.rgb_channels + self.model.ir_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, H, W), cond)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         # predicting x_0
        #         x_self_cond = self.model_predictions(x, t, cond)
        #         x_self_cond = x_self_cond.pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step
        noise_pred, fused_img = self.model(x[:,:3,...], x[:,-1:,...], t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        fusion_loss = self.fusion_lossfn(fused_img, x_start[:,:3,...], x_start[:,-1:,...])
        return {"noise_loss": 0.1 * F.l1_loss(noise_pred, target), "fusion_loss": fusion_loss}
              


    def forward(self, x_start, cond, *args, **kwargs):
        assert x_start.shape[1] == self.model.rgb_channels + self.model.ir_channels, \
            f"your input rgbt img must have {self.model.rgb_channels + self.model.ir_channels} channels"
        assert cond.shape[1] == self.model.cond_emb_channels, \
            f"condition should have {self.model.cond_emb_channels} channels"

        device = self.device
        x_start, cond  = x_start.to(device), cond.to(device)

        b, c, h, w = x_start.shape

        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x_start, times, cond, *args, **kwargs)


class SegDiTbackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.diff = SegDiff(
            model=DiT(
                patch_size=cfg.MODEL.DIT.PATCH_SIZE,
                rgb_channels=3,
                ir_channels=1,
                cond_emb_channels=5,
                hidden_size=cfg.MODEL.DIT.HIDDEN_SIZE,
                depth=cfg.MODEL.DIT.DEPTH,
                num_heads=cfg.MODEL.DIT.NUM_HEADS,
                mlp_ratio=cfg.MODEL.DIT.MLP_RATIO,
                cross_layers=0,
            ),
            timesteps=cfg.MODEL.DIT.TIMESTEPS,
            objective=cfg.MODEL.DIT.OBJECTIVE,
            sampling_timesteps=getattr(cfg.MODEL.DIT, "SAMPLING_STEPS", None),
            ddim_sampling_eta=getattr(cfg.MODEL.DIT, "DDIM_ETA", 1.0),
        )

    @torch.no_grad()
    def _compute_rgb_gradient(self, rgb):
        B, _, H, W = rgb.shape
        device = rgb.device

        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32, device=device).view(1,1,3,3)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)


        grad_x = F.conv2d(rgb, kx.repeat(3,1,1,1), padding=1, groups=3)
        grad_y = F.conv2d(rgb, ky.repeat(3,1,1,1), padding=1, groups=3)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad = grad.mean(dim=1, keepdim=True)

        gmin = grad.amin(dim=(2,3), keepdim=True)
        gmax = grad.amax(dim=(2,3), keepdim=True)
        grad = (grad - gmin) / (gmax - gmin + 1e-8)
        return (grad - 0.5) / 0.5

    def forward(self, x: torch.Tensor, infer: bool = False):
        assert x.dim() == 4 and x.size(1) == 4, "input must be [B,4,H,W] (RGB3 + IR1)"
        x = self.normalize_rgbt(x)
        rgb, ir = x[:, :3], x[:, 3:4]

        grad = self._compute_rgb_gradient(rgb)
        cond = torch.cat([rgb, ir, grad], dim=1)

        if not infer:
            return self.diff(x, cond)
        else:
            _, fused = self.diff.sample(cond)
            return fused
    
    def normalize_rgbt(self, img):
        mean = torch.tensor([119.2332, 118.6066, 116.0577, 78.7621], device=img.device).view(1,4,1,1)
        std  = torch.tensor([44.7161, 44.5107, 46.2892, 48.9237], device=img.device).view(1,4,1,1)
        return (img - mean) / std

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()

def auto_pad_to_multiple(img, multiple=32, mode="reflect"):
    """Pad image tensor (B,C,H,W) or (C,H,W) to the nearest multiple of `multiple`."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    B, C, H, W = img.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode=mode)
    if squeeze_back:
        img = img.squeeze(0)
    return img, (H, W), (pad_h, pad_w)


def custom_collate(batch, multiple=32):
    images, orig_sizes, pad_info = [], [], []
    for item in batch:
        img = item["image"]
        img, orig_hw, pad_hw = auto_pad_to_multiple(img, multiple=multiple)
        images.append(img)
        orig_sizes.append(orig_hw)
        pad_info.append(pad_hw)
    images = torch.stack(images, dim=0)
    return {"image": images, "orig_hw": orig_sizes, "pad_hw": pad_info}

def collate_fn_32(batch):
    return custom_collate(batch, multiple=32)


def setup_ddp():
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def main(train=True):
    rank, local_rank, world_size = setup_ddp()

    cfg = SimpleNamespace(
        DATASETS=SimpleNamespace(
            NAME="FMBdataset",
            DIR="./datasets/FMB_dataset",
            IMS_PER_BATCH=8,
            WORKERS_PER_GPU=4,
        ),
        MODEL=SimpleNamespace(
            SEM_SEG_HEAD=SimpleNamespace(
                IGNORE_VALUE=0,
                NUM_CLASSES=15,
            ),
            DIT=SimpleNamespace(
                INPUT_SIZE=(320, 416),
                PATCH_SIZE=8,
                IN_CHANNELS=4,
                COND_CHANNELS=4,
                COND_EMB_CHANNELS=1,
                HIDDEN_SIZE=768,
                DEPTH=18,
                NUM_HEADS=8,
                MLP_RATIO=4.0,
                TIMESTEPS=100,
                OBJECTIVE="pred_noise",
                SAMPLING_STEPS=1,
            ),
        ),
        INPUT=SimpleNamespace(
            MIN_SIZE_TRAIN=[int(x * 0.1 * 600) for x in range(5, 21)],
            MIN_SIZE_TRAIN_SAMPLING="choice",
            MIN_SIZE_TEST=480,
            MAX_SIZE_TRAIN=1920,
            MAX_SIZE_TEST=960,
            CROP=SimpleNamespace(
                ENABLED=True,
                TYPE="absolute",
                SIZE=(300, 400),
                SINGLE_CATEGORY_MAX_AREA=1.0,
            ),
            COLOR_AUG_SSD=True,
            SIZE_DIVISIBILITY=-1,
            FORMAT="RGBT",
            DATASET_MAPPER_NAME="mask_former_semantic",
        ),
        TRAIN=SimpleNamespace(
            DATA_DIR="./datasets/FMB_dataset",
            BATCH_SIZE=3,
            NUM_WORKERS=4,
            LR=1e-4,
            EPOCHS=200,
            DEVICE=f"cuda:{local_rank}",
            SAVE_DIR="./pretrained_model/SegDiT_ckpt/",
            RESUME=True,
        ),
    )

    if rank == 0:
        os.makedirs(cfg.TRAIN.SAVE_DIR, exist_ok=True)
    dist.barrier()

    train_set = FMB_dataset(cfg.TRAIN.DATA_DIR, cfg, split="train")
    val_set = FMB_dataset(cfg.TRAIN.DATA_DIR, cfg, split="test")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                              sampler=train_sampler, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn_32)
    val_loader = DataLoader(val_set, batch_size=1,
                            sampler=val_sampler, num_workers=cfg.TRAIN.NUM_WORKERS,
                            collate_fn=collate_fn_32)

    print(f"Rank {rank} | Dataloader ready | Total {len(train_loader)} batches")

    model = SegDiTbackbone(cfg).to(cfg.TRAIN.DEVICE)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)

    # --- Resume ---
    start_epoch = 0
    best_loss = float("inf")
    ckpt_path = os.path.join(cfg.TRAIN.SAVE_DIR, "segdit_best.pth")
    last_ckpt_path = os.path.join(cfg.TRAIN.SAVE_DIR, "segdit_last.pth")

    if cfg.TRAIN.RESUME and os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path, map_location=f"cuda:{local_rank}")
        model.module.load_state_dict(checkpoint["model_state"], strict=False)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if train:
            model.train()
            train_sampler.set_epoch(epoch)
            if rank == 0:
                pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS}", ncols=100)

            for batch in train_loader:
                x = batch["image"].to(cfg.TRAIN.DEVICE)
                loss = model(x, infer=False)
                loss_a = loss["noise_loss"] + loss["fusion_loss"]
                noise_loss = loss["noise_loss"] 
                fusion_loss = loss["fusion_loss"]
                optimizer.zero_grad()
                loss_a.backward()
                optimizer.step()

                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{noise_loss:.3f}, {fusion_loss:.3f}")
            if rank == 0:
                pbar.close()

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            if rank == 0:
                pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS}", ncols=100)
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(cfg.TRAIN.DEVICE)
                    loss = model(x, infer=False)
                    loss = loss["noise_loss"] + loss["fusion_loss"]
                    val_loss += loss.item() * x.size(0)

                    if rank == 0:
                        pbar.update(1)
                        pbar.set_postfix(loss=f"{loss}")

            if rank == 0:
                pbar.close()

            val_loss_tensor = torch.tensor(val_loss, device=cfg.TRAIN.DEVICE)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_avg = val_loss_tensor.item() / len(val_set)
            print(f"val_loss_avg:{val_loss_avg}")

            if rank == 0:
                print(f"[Epoch {epoch+1}] Val Loss: {val_loss_avg:.6f} | Best: {best_loss:.6f}")

                # Save last
                torch.save({
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_loss": best_loss
                }, last_ckpt_path)

                # Save best
                if val_loss_avg < best_loss:
                    best_loss = val_loss_avg
                    torch.save(model.module.state_dict(), ckpt_path)
                    print(f"New best checkpoint saved (val loss {best_loss:.6f})")

        else:
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    if idx == 0:
                        x = batch["image"].to(cfg.TRAIN.DEVICE)
                        fused_img = model(x, infer=True)

                        rgb = x[:,:3]/255.0
                        ir = x[:,-1:].repeat(1,3,1,1)/255.0
                        fused_vis = (fused_img.clamp(-1,1)+1)/2

                        from torchvision.utils import save_image
                        save_image(rgb, './images/rgb.png')
                        save_image(ir, './images/ir.png')
                        save_image(fused_vis, './images/fused.png')
                        if rank == 0:
                            print("Saved RGB, IR, Fused images.")

    cleanup_ddp()

    # model = FusionUNet(
    #     rgb_channels=3,
    #     ir_channels=1,
    #     cond_emb_channels=1,
    #     base_channels=64,
    #     time_emb_dim=64,
    #     channel_mults=(1,2,4),
    # )
    # n_params_M = count_params_in_m(model)
    # print(f"DiT params = {n_params_M:.2f} M")
    # exit(0)


def count_params_in_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

if __name__ == "__main__":
    main(True)
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 /home/tianjiayun/code/MMDiT/PEAFusion_DiT/models/mask2former/backbone/DiT.py
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 /home/tianjiayun/code/MMDiT/PEAFusion_DiT/models/mask2former/backbone/DiT.py