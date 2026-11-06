#!/usr/bin/env python3
"""
Phase B Evaluation Script: Rigorous benchmark of all candidate checkpoints
on the 129-identity validation split (387 condition pairs, seed 2026).

Vectorized execution:
- Batch-level inference with torch.inference_mode()
- Vectorized SSIM, MS-SSIM, and LPIPS over batch
- Vectorized MSE, MAE, PSNR
- CIEDE2000 per sample via skimage.color.deltaE_ciede2000
- Multi-threaded CPU optimization (torch.set_num_threads(4))

Evaluates:
- Epoch 1 (Restormer_RDB_epoch_001.pt)
- Epoch 2 (Restormer_RDB_epoch_002.pt)
- Epoch 3 (Restormer_RDB_epoch_003.pt / best_restormer.pt)
"""

import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TF
from torchvision.transforms import InterpolationMode
from pytorch_msssim import ms_ssim, ssim
import skimage.color
import lpips
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Multi-thread optimization for 4-core physical CPU
torch.set_num_threads(4)

backend_root = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_root))

from app.core.model import create_model, count_trainable_parameters
from train_or_eval import build_dataset_pairs, CVD_TYPES, CONDITION_TO_ID, find_default_data_root

DEVICE = torch.device("cpu")


class ValidationDataset(Dataset):
    def __init__(self, pairs: List[Dict], image_size: int = 128):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        orig_img = Image.open(item["original_path"]).convert("RGB")
        targ_img = Image.open(item["target_path"]).convert("RGB")

        orig_resized = TF.functional.resize(
            orig_img, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC
        )
        targ_resized = TF.functional.resize(
            targ_img, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC
        )

        orig_tensor = TF.functional.to_tensor(orig_resized)
        targ_tensor = TF.functional.to_tensor(targ_resized)

        return {
            "image": orig_tensor,
            "target": targ_tensor,
            "condition_id": torch.tensor(item["condition_id"], dtype=torch.long),
            "condition": item["condition"],
            "image_id": item["image_id"],
        }


def evaluate_checkpoint_on_val(
    checkpoint_path: Path,
    val_loader: DataLoader,
    lpips_fn: lpips.LPIPS,
) -> Dict[str, Any]:
    print(f"\nEvaluating Checkpoint: {checkpoint_path.name}")
    file_size_bytes = checkpoint_path.stat().st_size

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("architecture", "Restormer_RDB")
    epoch = ckpt.get("epoch", "unknown")
    val_loss_stored = ckpt.get("val_loss", None)

    clean_sd = {}
    for k, v in raw_sd.items():
        if "total_ops" in k or "total_params" in k:
            continue
        clean_k = k
        for pfx in ("_orig_mod.", "module."):
            if clean_k.startswith(pfx):
                clean_k = clean_k[len(pfx):]
        clean_sd[clean_k] = v

    model = create_model(arch).to(DEVICE)
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    param_count = count_trainable_parameters(model)

    condition_metrics = {c: [] for c in CVD_TYPES}
    all_metrics = []
    inference_times_ms = []

    # Warmup
    dummy_x = torch.zeros(1, 3, 128, 128, device=DEVICE)
    dummy_c = torch.tensor([1], device=DEVICE)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy_x, dummy_c)

    # Evaluation loop
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc=f"Val Eval ({checkpoint_path.stem})"):
            imgs = batch["image"].to(DEVICE)
            targets = batch["target"].to(DEVICE)
            cond_ids = batch["condition_id"].to(DEVICE)
            cond_names = batch["condition"]
            batch_sz = imgs.size(0)

            t0 = time.perf_counter()
            preds = model(imgs, cond_ids)
            t1 = time.perf_counter()
            per_img_inf_time = ((t1 - t0) * 1000.0) / batch_sz
            inference_times_ms.extend([per_img_inf_time] * batch_sz)

            preds_c = preds.clamp(0.0, 1.0)
            targets_c = targets.clamp(0.0, 1.0)

            # Vectorized MAE & MSE
            diff = preds_c - targets_c
            mae_batch = diff.abs().mean(dim=[1, 2, 3]).cpu().tolist()
            mse_batch = (diff ** 2).mean(dim=[1, 2, 3]).cpu().tolist()
            psnr_batch = [10.0 * np.log10(1.0 / max(m, 1e-10)) for m in mse_batch]

            # Vectorized SSIM & MS-SSIM
            ssim_batch = ssim(preds_c, targets_c, data_range=1.0, size_average=False).cpu().tolist()
            ms_ssim_batch = ms_ssim(preds_c, targets_c, data_range=1.0, size_average=False, win_size=7).cpu().tolist()

            # Vectorized LPIPS
            lpips_res = lpips_fn(preds_c * 2.0 - 1.0, targets_c * 2.0 - 1.0).squeeze().cpu().tolist()
            if isinstance(lpips_res, float):
                lpips_batch = [lpips_res]
            else:
                lpips_batch = lpips_res

            # CIEDE2000 per sample
            preds_np = preds_c.permute(0, 2, 3, 1).cpu().numpy()
            targets_np = targets_c.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(batch_sz):
                p_lab = skimage.color.rgb2lab(preds_np[i])
                t_lab = skimage.color.rgb2lab(targets_np[i])
                ciede = float(np.mean(skimage.color.deltaE_ciede2000(p_lab, t_lab)))

                m = {
                    "psnr": float(psnr_batch[i]),
                    "ssim": float(ssim_batch[i]),
                    "ms_ssim": float(ms_ssim_batch[i]),
                    "mae": float(mae_batch[i]),
                    "mse": float(mse_batch[i]),
                    "lpips": float(lpips_batch[i]),
                    "ciede2000": ciede,
                }
                c_name = cond_names[i]
                condition_metrics[c_name].append(m)
                all_metrics.append(m)

    def aggregate(m_list):
        return {
            "psnr": float(np.mean([m["psnr"] for m in m_list])),
            "ssim": float(np.mean([m["ssim"] for m in m_list])),
            "ms_ssim": float(np.mean([m["ms_ssim"] for m in m_list])),
            "mae": float(np.mean([m["mae"] for m in m_list])),
            "mse": float(np.mean([m["mse"] for m in m_list])),
            "lpips": float(np.mean([m["lpips"] for m in m_list])),
            "ciede2000": float(np.mean([m["ciede2000"] for m in m_list])),
            "count": len(m_list),
        }

    results = {
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "architecture": arch,
        "epoch": epoch,
        "stored_val_loss": val_loss_stored,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "parameter_count": param_count,
        "latency_ms_per_image": float(np.mean(inference_times_ms)),
        "Overall": aggregate(all_metrics),
    }

    for c in CVD_TYPES:
        results[c] = aggregate(condition_metrics[c])

    return results


def main():
    print("=" * 80)
    print("PHASE B: CANDIDATE CHECKPOINT BENCHMARK ON VALIDATION SPLIT")
    print(f"Device: {DEVICE} (4 threads) | Target Resolution: 128x128")
    print("=" * 80)

    data_root = find_default_data_root()
    train_pairs, val_pairs, test_pairs = build_dataset_pairs(data_root, seed=2026)
    print(f"Validation Identities: 129 | Validation Pairs: {len(val_pairs)}")

    val_loader = DataLoader(
        ValidationDataset(val_pairs, image_size=128),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    print("Initializing AlexNet LPIPS perceptual loss model...")
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    weights_dir = backend_root / "weights"
    checkpoints = [
        weights_dir / "checkpoints" / "Restormer_RDB_epoch_001.pt",
        weights_dir / "checkpoints" / "Restormer_RDB_epoch_002.pt",
        weights_dir / "checkpoints" / "Restormer_RDB_epoch_003.pt",
        weights_dir / "best_restormer.pt",
    ]

    all_results = []
    for ckpt_path in checkpoints:
        if not ckpt_path.is_file():
            print(f"WARNING: Checkpoint {ckpt_path} not found. Skipping.")
            continue
        res = evaluate_checkpoint_on_val(ckpt_path, val_loader, lpips_fn)
        all_results.append(res)

    out_json = backend_root / "validation_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nValidation benchmark results saved to: {out_json}")

    print("\n" + "=" * 94)
    print("VALIDATION BENCHMARK SUMMARY TABLE (129 Identities, 387 Condition Pairs)")
    print("=" * 94)
    header = f"{'Checkpoint':<28} | {'Epoch':<5} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'MS-SSIM':<7} | {'MAE':<7} | {'LPIPS':<7} | {'DeltaE':<7} | {'CPU Lat':<7}"
    print(header)
    print("-" * 94)
    for r in all_results:
        ov = r["Overall"]
        name = r["checkpoint_file"]
        ep = str(r["epoch"])
        psnr = f"{ov['psnr']:.2f}"
        ssim_v = f"{ov['ssim']:.4f}"
        ms_ssim_v = f"{ov['ms_ssim']:.4f}"
        mae_v = f"{ov['mae']:.4f}"
        lpips_v = f"{ov['lpips']:.4f}"
        de_v = f"{ov['ciede2000']:.2f}"
        lat = f"{r['latency_ms_per_image']:.1f}ms"
        print(f"{name:<28} | {ep:<5} | {psnr:<9} | {ssim_v:<7} | {ms_ssim_v:<7} | {mae_v:<7} | {lpips_v:<7} | {de_v:<7} | {lat:<7}")

    print("=" * 94)

    # Condition-level breakdown for winning checkpoint
    best_res = all_results[-1]  # best_restormer / epoch 3
    print(f"\nDetailed Condition Breakdown for {best_res['checkpoint_file']} (Epoch {best_res['epoch']}):")
    print(f"{'Condition':<15} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'MS-SSIM':<7} | {'MAE':<7} | {'LPIPS':<7} | {'DeltaE':<7}")
    print("-" * 75)
    for c in CVD_TYPES:
        m = best_res[c]
        print(f"{c:<15} | {m['psnr']:<9.2f} | {m['ssim']:<7.4f} | {m['ms_ssim']:<7.4f} | {m['mae']:<7.4f} | {m['lpips']:<7.4f} | {m['ciede2000']:<7.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
