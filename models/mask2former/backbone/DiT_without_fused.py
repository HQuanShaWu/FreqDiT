import sys, os
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
from timm.models.layers.helpers import to_2tuple
import numpy as np
from dataloaders.FMB_dataset import FMB_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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
    
class FMCmodule(nn.Module):
    def __init__(self, hidden_dim: int, order: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.order = order

        self.t2p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        self.t2cut = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid()
        )

        self._init_weights()

    @staticmethod
    def _make_radial_grid(H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        rr = torch.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)
        rr = rr / (rr.max() + 1e-8)
        return rr

    def _butterworth(self, rr, cutoff, order=4, mode='low'):
        if mode == 'low':
            return 1.0 / (1.0 + (rr / cutoff) ** (2 * order))
        elif mode == 'high':
            return 1.0 - 1.0 / (1.0 + (rr / cutoff) ** (2 * order))
        else:
            raise ValueError("mode must be 'low' or 'high'")

    def _fft_filter(self, x, mask):
        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        Xf = Xf * mask
        x_filt = torch.fft.ifft2(torch.fft.ifftshift(Xf, dim=(-2, -1)), norm="ortho").real
        return x_filt

    def _init_weights(self):
        for layer in self.t2p:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if isinstance(self.t2p[-2], nn.Linear):
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
        if isinstance(self.t2cut[-2], nn.Linear):
            self.t2cut[-2].weight.data -= torch.abs(
                torch.randn_like(self.t2cut[-2].weight)
            ) * 0.1
            if self.t2cut[-2].bias is not None:
                self.t2cut[-2].bias.data.fill_(0.5)


    def forward(self, cond, t_emb, h=None):
        B, C, H, W = cond.shape
        device = cond.device

        p_t = self.t2p(t_emb).view(B, 1, 1, 1)
        f_cut = self.t2cut(t_emb)
        f_lp, f_hp = f_cut[:, :1], f_cut[:, 1:]

        rr = self._make_radial_grid(H, W, device)
        M_lp = self._butterworth(rr, f_lp.view(-1, 1, 1), self.order, mode="low")
        M_hp = self._butterworth(rr, f_hp.view(-1, 1, 1), self.order, mode="high")

        M_lp = M_lp.view(B, 1, H, W)
        M_hp = M_hp.view(B, 1, H, W)

        M_t = p_t * M_lp + (1 - p_t) * (M_hp - M_lp)
        cond_f = self._fft_filter(cond, M_t)

        if h is not None:
            cond_f = F.interpolate(cond_f, size=h.shape[-2:], mode='bilinear', align_corners=False)

        return cond_f


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
        patch_size=4,
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
        patch_size=4,
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
        self.rgbt_embedder = PatchEmbed(None, patch_size, rgb_channels+ir_channels, hidden_size)
        self.emb_embedder = PatchEmbed(None, patch_size, cond_emb_channels, hidden_size)
        self.t_embedder   = TimestepEmbedder(hidden_size)
        self.cond_filter = FMCmodule(hidden_dim=self.hidden)

        self.cross_blocks = nn.ModuleList([
            CrossModalBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(cross_layers)
        ])
        self.main_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth - cross_layers)
        ])

        self.noise_head = FinalLayer(hidden_size, patch_size, out_ch=rgb_channels + ir_channels)
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
        rgbt = torch.cat([rgb, ir], dim=1)
        rgbt_tok = self.rgbt_embedder(rgbt)
        patch_num = rgbt_tok.shape[1]

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=rgb.device).repeat(B)
        t_emb = self.t_embedder(t.long())

        if cond is not None:
            cond_filtered = self.cond_filter(cond, t_emb)
            cond_tok = self.emb_embedder(cond_filtered)
            tokens = torch.cat([rgbt_tok, cond_tok], dim=1)
        else:
            tokens = rgbt_tok

        pos = torch.from_numpy(get_2d_sincos_pos_embed(self.hidden, (H // self.patch, W // self.patch))).float().to(rgb.device)
        pos = pos.unsqueeze(0)
        if cond is not None:
            cond_pos = pos[:, :cond_tok.shape[1], :] if cond_tok.shape[1] <= pos.shape[1] else pos[:, :1, :].repeat(1, cond_tok.shape[1], 1)
            pos = torch.cat([pos, cond_pos], dim=1)
        tokens = tokens + pos

        h = tokens
        for blk in self.main_blocks:
            h = blk(h, t_emb)

        out_tokens = self.noise_head(h[:, :patch_num], t_emb)
        eps_hat = self.unpatchify(out_tokens, H, W, self.rgb_channels + self.ir_channels)

        return eps_hat


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

    def model_predictions(self, x, t, c, clip_x_start=False):
        model_output = self.model(x[:,:3,...], x[:,-1:,...], t, c)
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

        return ModelPrediction(pred_noise, x_start)

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
            preds = self.model_predictions(img, time_cond, cond,
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

        return img

    @torch.no_grad()
    def sample(self, cond):
        batch_size, device = cond.shape[0], self.device
        H, W = cond.shape[2], cond.shape[3]
        cond = cond.to(self.device)
        mask_channels = self.model.rgb_channels + self.model.ir_channels
        sample_fn = self.ddim_sample
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

        # predict and take gradient step
        noise_pred = self.model(x[:,:3,...], x[:,-1:,...], t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return F.mse_loss(noise_pred, target)
              


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
                cond_emb_channels=cfg.MODEL.DIT.COND_EMB_CHANNELS,
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

    @torch.no_grad()
    def _compute_gradient(self, x):
        B, C, H, W = x.shape
        device = x.device
        kx = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        grad_x = F.conv2d(x, kx.repeat(C, 1, 1, 1), padding=1, groups=C)
        grad_y = F.conv2d(x, ky.repeat(C, 1, 1, 1), padding=1, groups=C)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad = grad.mean(dim=1, keepdim=True)

        gmin = grad.amin(dim=(2, 3), keepdim=True)
        gmax = grad.amax(dim=(2, 3), keepdim=True)
        grad = (grad - gmin) / (gmax - gmin + 1e-8)
        return (grad - 0.5) / 0.5

    def forward(self, x: torch.Tensor, infer: bool = False):
        assert x.dim() == 4 and x.size(1) == 4, "input must be [B,4,H,W] (RGB3 + IR1)"
        x = self.normalize_rgbt(x)
        rgb, ir = x[:, :3], x[:, 3:4]

        grad_rgb = self._compute_rgb_gradient(rgb)
        grad_ir  = self._compute_gradient(ir)
        cond = torch.cat([rgb, ir, grad_rgb, grad_ir], dim=1)

        if not infer:
            return self.diff(x, cond)
        else:
            img = self.diff.sample(cond)
            return img
    
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
                COND_EMB_CHANNELS=6,
                HIDDEN_SIZE=128,
                DEPTH=3,
                NUM_HEADS=8,
                MLP_RATIO=4.0,
                TIMESTEPS=1000,
                OBJECTIVE="pred_noise",
                SAMPLING_STEPS=1000,
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
            BATCH_SIZE=4,
            NUM_WORKERS=4,
            LR=3e-4,
            EPOCHS=200,
            DEVICE=f"cuda:{local_rank}",
            SAVE_DIR="./pretrained_model/SegDiT_wo_fusion",
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
        model.train()
        train_sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS}", ncols=100)

        for batch in train_loader:
            x = batch["image"].to(cfg.TRAIN.DEVICE)
            loss = model(x, infer=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss:.3f}")
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
                val_loss += loss * x.size(0)

                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss}")

        if rank == 0:
            pbar.close()

        val_loss_tensor = val_loss.clone().detach()
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

    cleanup_ddp()

if __name__ == "__main__":
    main(True)
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 /home/tianjiayun/code/MMDiT/PEAFusion_DiT/models/mask2former/backbone/DiT_without_fused.py
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 /home/tianjiayun/code/MMDiT/PEAFusion_DiT/models/mask2former/backbone/DiT_without_fused.py