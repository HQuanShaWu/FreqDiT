#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FMC Frequency Probe (STRICT cfg mode for SegDiTbackbone) — full version

- Uses your exact cfg structure (DATASETS/MODEL/INPUT/TRAIN)
- Builds SegDiTbackbone(cfg) and FMB_dataset(cfg=cfg, data_dir=..., split=...)
- Loads SegDiTbackbone checkpoint (strict=False)
- Runs reverse process for T=cfg.MODEL.DIT.TIMESTEPS
- Computes robust frequency metrics with Hann window:
    * spectral centroid (0~1) — MAIN curve
    * multi-band energy (time × radius heatmap) + saves npy
- Compares FMC ON vs OFF (by toggling cond_filter), with progress bars
- Logs TensorBoard curves; saves PNG/NPY; optional x_start snapshots

Run from repo root:
  python -m models.mask2former.backbone.fmc_frequency_probe_fromcfg_full \
    --ckpt ./pretrained_model/SegDiT_ckpt/segdit_last.pth \
    --logdir ./runs/frequency_probe
"""

import os, sys, argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs): return x  # fallback

# ---------- path setup ----------
_THIS = os.path.abspath(__file__)
_CUR = os.path.dirname(_THIS)
def _add_project_root_for_models():
    p = _CUR
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "models")):
            if p not in sys.path: sys.path.insert(0, p)
            return p
        p = os.path.dirname(p)
    return None
_PROJECT_ROOT = _add_project_root_for_models()

try:
    from models.mask2former.backbone.DiT_without_fused import SegDiTbackbone
except Exception as e_pkg:
    try:
        from DiT_without_fused import SegDiTbackbone
    except Exception as e_local:
        raise ImportError(f"Cannot import SegDiTbackbone. package err: {e_pkg}  local err: {e_local}")

# dataset class (best-effort lookup)
FMB_dataset = None
for cand in ["dataloaders.FMB_dataset","models.mask2former.backbone.dataloaders.FMB_dataset","datasets.FMB_dataset"]:
    try:
        module = __import__(cand, fromlist=['FMB_dataset'])
        FMB_dataset = getattr(module, 'FMB_dataset', None)
        if FMB_dataset is not None: break
    except Exception: pass

# ---------- utils ----------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 if x.min() < 0 else x

def _save_rgb(path: str, tensor_bchw: torch.Tensor):
    # tensor: (1,3,H,W) in [-1,1] or [0,1]
    x = _to01(tensor_bchw).clamp(0,1)[0].permute(1,2,0).detach().cpu().numpy()
    plt.imsave(path, x)

# ---------- build your cfg exactly ----------
def build_user_cfg(local_rank: int, override_dataset_dir: str = None, override_train_data_dir: str = None):
    DATASETS_DIR = override_dataset_dir or "./datasets/FMB_dataset"
    TRAIN_DATA_DIR = override_train_data_dir or "./datasets/FMB_dataset"

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
    return cfg

# ---------- build model from cfg and load ckpt ----------
def load_backbone_from_cfg_and_ckpt(cfg, ckpt_path, device):
    print(cfg.MODEL.DIT.COND_EMB_CHANNELS)
    net = SegDiTbackbone(cfg).to(device).eval()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        for k in ["model_state","state_dict","ema_state_dict","net"]:
            if k in blob and isinstance(blob[k], dict):
                sd = blob[k]; break
        else:
            sd = {k:v for k,v in blob.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError("Unsupported checkpoint format")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[Load] Missing {len(missing)}, Unexpected {len(unexpected)}")
    if missing: print("  Missing e.g.:", missing[:5])
    if unexpected: print("  Unexpected e.g.:", unexpected[:5])
    return net

# ---------- dataloader using cfg ----------
def _collate_detectron2_safe(batch):
    """Return the first item of the batch; avoids collating detectron2 Instances."""
    return batch[0]

def build_dataloader_with_cfg(cfg, split='val'):
    if FMB_dataset is None:
        raise RuntimeError("FMB_dataset not found in your repo.")
    dataset = FMB_dataset(cfg=cfg, data_dir=cfg.DATASETS.DIR, split=split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=_collate_detectron2_safe)
    return loader

def pick_input_from_sample(sample):
    import numpy as _np
    if isinstance(sample, dict):
        x = None
        for k in ["image","img","rgbt","rgb_ir_4ch","rgb_ir","x4"]:
            if k in sample:
                x = sample[k]; break
        if x is not None:
            if torch.is_tensor(x):
                if x.dim()==4 and x.size(1)==4:
                    return x.float()
                if x.dim()==3 and x.size(0) in (3,4):
                    return x.unsqueeze(0).float()
            if isinstance(x, _np.ndarray) and x.ndim==3 and x.shape[-1] in (3,4):
                x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
                return x.float()
        # fallback by rgb/ir
        rgb, ir = sample.get("rgb"), sample.get("ir")
        if rgb is None or ir is None:
            raise ValueError("Need a 4ch image (under keys image/img/rgbt/...) or (rgb, ir).")
        if isinstance(rgb, _np.ndarray): rgb = torch.from_numpy(rgb)
        if isinstance(ir, _np.ndarray): ir = torch.from_numpy(ir)
        if rgb.ndim==3 and rgb.shape[-1]==3: rgb = rgb.permute(2,0,1)
        if ir.ndim==3 and ir.shape[-1]==1: ir = ir.permute(2,0,1)
        if ir.ndim==2: ir = ir.unsqueeze(0)
        x = torch.cat([rgb, ir], dim=0).unsqueeze(0)
        return x.float()
    elif isinstance(sample, (list,tuple)):
        a = sample[0]
        if torch.is_tensor(a) and a.dim()==4 and a.size(1)==4:
            return a.float()
        rgb, ir = sample[0], sample[1]
        if isinstance(rgb, np.ndarray): rgb = torch.from_numpy(rgb)
        if isinstance(ir, np.ndarray): ir = torch.from_numpy(ir)
        if rgb.ndim==3 and rgb.shape[-1]==3: rgb = rgb.permute(2,0,1)
        if ir.ndim==3 and ir.shape[-1]==1: ir = ir.permute(2,0,1)
        if ir.ndim==2: ir = ir.unsqueeze(0)
        x = torch.cat([rgb, ir], dim=0).unsqueeze(0)
        return x.float()
    else:
        raise ValueError("Unknown sample format.")

# ---------- frequency metrics ----------
def _apply_hann_window(img: torch.Tensor) -> torch.Tensor:
    # img: (1, 1, H, W)
    B, C, H, W = img.shape
    win_h = torch.hann_window(H, periodic=True, device=img.device).view(1, 1, H, 1)
    win_w = torch.hann_window(W, periodic=True, device=img.device).view(1, 1, 1, W)
    return img * (win_h * win_w)

@torch.no_grad()
def spectral_centroid(img_rgb: torch.Tensor) -> float:
    """
    Robust single-value descriptor of frequency content: 0(low) ~ 1(high).
    img_rgb: (1,3,H,W) in [-1,1] or [0,1]
    """
    x01 = _to01(img_rgb)
    gray = (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]).to(torch.float32)
    gray = _apply_hann_window(gray)
    Xf = torch.fft.fftshift(torch.fft.fft2(gray.to(torch.float64), norm="ortho"), dim=(-2,-1))
    mag2 = (Xf.real**2 + Xf.imag**2).squeeze(0).squeeze(0)  # (H,W) float64
    H, W = mag2.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=mag2.device),
                            torch.arange(W, device=mag2.device), indexing="ij")
    rr = torch.sqrt((xx - W//2)**2 + (yy - H//2)**2) / (max(H, W)/2.0 + 1e-8)
    return float((mag2 * rr).sum() / (mag2.sum() + 1e-12))

@torch.no_grad()
def radial_band_energy_profile(img_rgb: torch.Tensor, num_bins: int = 20) -> torch.Tensor:
    """
    Returns (num_bins,) band energy ratio over radius bins (time × radius for heatmap).
    img_rgb: (1,3,H,W) in [-1,1] or [0,1]
    """
    x01 = _to01(img_rgb)
    gray = (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]).to(torch.float32)
    gray = _apply_hann_window(gray)
    Xf = torch.fft.fftshift(torch.fft.fft2(gray.to(torch.float64), norm="ortho"), dim=(-2,-1))
    mag2 = (Xf.real**2 + Xf.imag**2).squeeze(0).squeeze(0)  # (H,W)
    H, W = mag2.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=mag2.device),
                            torch.arange(W, device=mag2.device), indexing="ij")
    rr = torch.sqrt((xx - W//2)**2 + (yy - H//2)**2)
    rr = rr / (rr.max() + 1e-8)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=mag2.device)
    total = mag2.sum() + 1e-12
    band_energy = []
    for i in range(num_bins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        e = mag2[m].sum()
        band_energy.append(e / total)
    band_energy = torch.stack(band_energy, dim=0)  # (K,)
    return band_energy.to(torch.float32)

class IdentityCondFilter(nn.Module):
    def forward(self, cond, t_emb): return cond

# ---------- probe ----------
@torch.no_grad()
def run_probe(backbone, x4, cfg, logdir, num_bins=20, save_snapshots="", device="cuda"):
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    try:
        # enforce input size (must be divisible by PATCH_SIZE)
        target_size = (int(cfg.MODEL.DIT.INPUT_SIZE[0]), int(cfg.MODEL.DIT.INPUT_SIZE[1]))  # (H,W)
        p = int(cfg.MODEL.DIT.PATCH_SIZE)
        assert target_size[0] % p == 0 and target_size[1] % p == 0, \
            f"INPUT_SIZE {target_size} must be divisible by PATCH_SIZE={p}"
        if x4.shape[-2:] != target_size:
            x4 = F.interpolate(x4, size=target_size, mode="bilinear", align_corners=False)

        def make_cond(x4):
            # x4: (B, 4, H, W) -> assume [R,G,B,IR]
            x_norm = backbone.normalize_rgbt(x4)
            rgb_n, ir_n = x_norm[:, :3], x_norm[:, 3:4]

            # Sobel kernels
            kx = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            ky = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

            # ---- grad on normalized RGB (per-channel depthwise, then aggregate to 1 channel) ----
            grad_x_rgb = F.conv2d(rgb_n, kx.repeat(3, 1, 1, 1), padding=1, groups=3)
            grad_y_rgb = F.conv2d(rgb_n, ky.repeat(3, 1, 1, 1), padding=1, groups=3)
            grad_mag_rgb_c = torch.sqrt(grad_x_rgb**2 + grad_y_rgb**2 + 1e-8)  # (B,3,H,W)
            grad_rgb = grad_mag_rgb_c.mean(dim=1, keepdim=True)                # (B,1,H,W)

            # ---- grad on normalized IR (single channel) ----
            grad_x_ir = F.conv2d(ir_n, kx, padding=1)
            grad_y_ir = F.conv2d(ir_n, ky, padding=1)
            grad_ir = torch.sqrt(grad_x_ir**2 + grad_y_ir**2 + 1e-8)           # (B,1,H,W)

            # cond: rgb, ir, grad_rgb, grad_ir  -> (B, 3+1+1+1=6, H, W)
            cond = torch.cat([rgb_n, ir_n, grad_rgb, grad_ir], dim=1)
            return x_norm, cond

        T = int(cfg.MODEL.DIT.TIMESTEPS)
        orig_filter = getattr(backbone.diff.model, "cond_filter", None)

        def iterate(fmc_on: bool, K: int = 20):
            # toggle FMC
            if hasattr(backbone.diff.model, "cond_filter"):
                if fmc_on:
                    backbone.diff.model.cond_filter = orig_filter
                else:
                    backbone.diff.model.cond_filter = IdentityCondFilter()
            else:
                print("[Warn] cond_filter not found; FMC toggle skipped.")

            x_norm, cond = make_cond(x4.to(device))
            B, C, H, W = x_norm.shape
            img = torch.randn((B, C, H, W), device=device)

            centroid_over_time = torch.zeros((T,), dtype=torch.float32, device="cpu")
            bands_over_time = torch.zeros((T, K), dtype=torch.float32, device="cpu")

            save_steps = set()
            if isinstance(save_snapshots, str) and len(save_snapshots.strip())>0:
                for s in save_snapshots.split(","):
                    try:
                        save_steps.add(int(s.strip()))
                    except Exception:
                        pass

            bar = tqdm(range(T - 1, -1, -1), desc=("FMC ON" if fmc_on else "FMC OFF"), dynamic_ncols=True)
            img_prev = None
            for step_idx, t in enumerate(bar):
                tt = torch.full((B,), t, device=device, dtype=torch.long)

                preds = backbone.diff.model_predictions(img, tt, cond, clip_x_start=False)
                x_start = preds.pred_x_start
                pred_noise = preds.pred_noise

                # --- DDIM deterministic update (保持和训练一致) ---
                alpha = backbone.diff.alphas_cumprod[t]
                alpha_prev = backbone.diff.alphas_cumprod[max(t - 1, 0)]
                sigma = backbone.diff.ddim_sampling_eta * ((1 - alpha / alpha_prev) * (1 - alpha_prev) / (1 - alpha)).sqrt()
                c = (1 - alpha_prev - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img_next = x_start * alpha_prev.sqrt() + c * pred_noise + sigma * noise

                # --- Δx 计算 ---
                if img_prev is not None:
                    delta = img_next - img
                    delta_rgb = delta[:, :3]
                    centroid = spectral_centroid(delta_rgb[0:1])
                    bands = radial_band_energy_profile(delta_rgb[0:1], num_bins=K)
                    centroid_over_time[T - 1 - step_idx] = centroid
                    bands_over_time[T - 1 - step_idx] = bands.cpu()

                    bar.set_postfix_str(f"t={t} Δx centroid={centroid:.4f}")

                img_prev = img.clone()
                img = img_next

                if t in save_steps:
                    save_dir = os.path.join(logdir, "snapshots_FMC" if fmc_on else "snapshots_noFMC")
                    os.makedirs(save_dir, exist_ok=True)
                    _save_rgb(os.path.join(save_dir, f"t_{t:04d}.png"), img[:, :3])

            return centroid_over_time, bands_over_time

        print(f"[Probe] T={T}  FMC ON")
        cen_on, bands_on = iterate(True,  K=num_bins)
        print("[Probe] FMC OFF")
        cen_off, bands_off = iterate(False, K=num_bins)

        # write scalars
        x = np.arange(T - 1, -1, -1)
        for i, t in enumerate(x):
            SummaryWriter.add_scalars  # keep static analyzer happy
            writer.add_scalars("spectral_centroid", {"with_FMC": float(cen_on[i]), "without_FMC": float(cen_off[i])}, global_step=int(t))

        # save npy
        np.save(os.path.join(logdir, "centroid_with_FMC.npy"),  cen_on.cpu().numpy())
        np.save(os.path.join(logdir, "centroid_without_FMC.npy"), cen_off.cpu().numpy())
        np.save(os.path.join(logdir, "bands_with_FMC.npy"),  bands_on.cpu().numpy())
        np.save(os.path.join(logdir, "bands_without_FMC.npy"), bands_off.cpu().numpy())
        np.save(os.path.join(logdir, "timesteps.npy"), x)

        # plot centroid curves
        plt.figure(figsize=(10,4))
        plt.plot(x, cen_off.cpu().numpy(), label="Without FMC")
        plt.plot(x, cen_on.cpu().numpy(),  label="With FMC")
        plt.xlabel("Timestep (t)"); plt.ylabel("Spectral centroid (0~1)")
        plt.title("Frequency evolution (spectral centroid)")
        plt.legend(); plt.tight_layout()
        fig_path1 = os.path.join(logdir, "curve_spectral_centroid.png")
        plt.savefig(fig_path1, dpi=200); plt.close()

        # plot heatmaps
        def _plot_heatmap(bands, title, path):
            plt.figure(figsize=(10,4))
            plt.imshow(bands.cpu().numpy().T, aspect="auto", origin="lower")
            plt.colorbar(); plt.title(title)
            plt.xlabel("time step (t from T→0)"); plt.ylabel("radial bin")
            plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

        _plot_heatmap(bands_on,  "With FMC: band energy (time × radius)", os.path.join(logdir, "heatmap_with_FMC.png"))
        _plot_heatmap(bands_off, "Without FMC: band energy (time × radius)", os.path.join(logdir, "heatmap_without_FMC.png"))

        # TB images
        try:
            img = plt.imread(fig_path1); writer.add_image("plots/spectral_centroid", torch.from_numpy(img).permute(2,0,1), global_step=0)
        except Exception: pass

        print(f"[Done] Logged to {logdir}")
        print(f"  Saved figure: {fig_path1}")
    finally:
        writer.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--logdir", type=str, default="./runs/frequency_probe")
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--seed", type=int, default=42)
    # dataset overrides
    ap.add_argument("--datasets_dir", type=str, default=None, help="Override cfg.DATASETS.DIR")
    ap.add_argument("--train_data_dir", type=str, default=None, help="Override cfg.TRAIN.DATA_DIR")
    ap.add_argument("--split", type=str, default="val")
    # single image fallback
    ap.add_argument("--use_single_image", type=str, default=None, help="Path to one 4ch .npy/.pt/.pth instead of dataloader")
    # metrics
    ap.add_argument("--num_bins", type=int, default=20, help="Number of radial bins for heatmap")
    ap.add_argument("--save_snapshots", type=str, default="100,80,60,40,20,0", help="Comma-separated timesteps to save x_start RGB (optional)")
    args = ap.parse_args()

    set_seed(args.seed)
    # infer local_rank from --device
    local_rank = 0
    if isinstance(args.device, str) and args.device.startswith("cuda:"):
        try:
            local_rank = int(args.device.split(":")[1])
        except Exception:
            local_rank = 0

    cfg = build_user_cfg(local_rank, override_dataset_dir=args.datasets_dir, override_train_data_dir=args.train_data_dir)
    device = args.device
    backbone = load_backbone_from_cfg_and_ckpt(cfg, args.ckpt, device)

    if args.use_single_image is None:
        loader = build_dataloader_with_cfg(cfg, split=args.split)
        batch = next(iter(loader))
        x4 = pick_input_from_sample(batch).to(device)
    else:
        p = args.use_single_image
        if p.endswith(".npy"):
            arr = np.load(p); ten = torch.from_numpy(arr)
        else:
            ten = torch.load(p, map_location="cpu")
        if ten.ndim==3 and ten.shape[0]==4: ten = ten.unsqueeze(0)
        elif ten.ndim==3 and ten.shape[-1]==4: ten = ten.permute(2,0,1).unsqueeze(0)
        elif ten.ndim==4 and ten.shape[1]==4: pass
        else: raise ValueError(f"Unsupported x4 shape: {tuple(ten.shape)}")
        x4 = ten.float().to(device)

    run_probe(backbone, x4, cfg, logdir=args.logdir, num_bins=args.num_bins, save_snapshots=args.save_snapshots, device=device)

if __name__ == "__main__":
    main()
