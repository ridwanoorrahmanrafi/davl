#!/usr/bin/env python3
"""
Phase D Evaluation Script: Single locked evaluation of the selected production reference model
(backend/weights/best_restormer.pt) on the held-out test split (128 identities, 384 pairs, seed 2026).

Computes:
- PSNR (dB)
- SSIM
- MS-SSIM
- MAE (L1 error)
- MSE
- LPIPS (AlexNet perceptual distance)
- CIEDE2000 (Delta E 00)
- Parameter count and model size
- Inference latency (ms per image on CPU)

Breakdowns:
- Overall (384 pairs)
- Protanopia (128 pairs)
- Deuteranopia (128 pairs)
- Tritanopia (128 pairs)
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


class TestDataset(Dataset):
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


def main():
    print("=" * 80)
    print("PHASE D: LOCKED PRODUCTION MODEL EVALUATION ON HELD-OUT TEST SPLIT")
    print(f"Device: {DEVICE} (4 threads) | Target Resolution: 128x128")
    print("=" * 80)

    # 1. Dataset split
    data_root = find_default_data_root()
    train_pairs, val_pairs, test_pairs = build_dataset_pairs(data_root, seed=2026)
    print(f"Test Identities: 128 | Test Condition Pairs: {len(test_pairs)}")

    test_loader = DataLoader(
        TestDataset(test_pairs, image_size=128),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # 2. LPIPS Model
    print("Initializing AlexNet LPIPS perceptual loss model...")
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    # 3. Load Production Reference Checkpoint
    checkpoint_path = backend_root / "weights" / "best_restormer.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Production reference weights not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("architecture", "Restormer_RDB")
    epoch = ckpt.get("epoch", 3)

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
    file_size_bytes = checkpoint_path.stat().st_size
    print(f"Model: {arch} (Epoch {epoch}) | Parameters: {param_count:,} | Size: {file_size_bytes / (1024*1024):.2f} MB")

    # Warmup
    dummy_x = torch.zeros(1, 3, 128, 128, device=DEVICE)
    dummy_c = torch.tensor([1], device=DEVICE)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy_x, dummy_c)

    condition_metrics = {c: [] for c in CVD_TYPES}
    all_metrics = []
    inference_times_ms = []

    print("\nExecuting evaluation over 384 test pairs...")
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Held-Out Test Eval"):
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

            diff = preds_c - targets_c
            mae_batch = diff.abs().mean(dim=[1, 2, 3]).cpu().tolist()
            mse_batch = (diff ** 2).mean(dim=[1, 2, 3]).cpu().tolist()
            psnr_batch = [10.0 * np.log10(1.0 / max(m, 1e-10)) for m in mse_batch]

            ssim_batch = ssim(preds_c, targets_c, data_range=1.0, size_average=False).cpu().tolist()
            ms_ssim_batch = ms_ssim(preds_c, targets_c, data_range=1.0, size_average=False, win_size=7).cpu().tolist()

            lpips_res = lpips_fn(preds_c * 2.0 - 1.0, targets_c * 2.0 - 1.0).squeeze().cpu().tolist()
            if isinstance(lpips_res, float):
                lpips_batch = [lpips_res]
            else:
                lpips_batch = lpips_res

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
        "architecture": arch,
        "epoch": epoch,
        "parameter_count": param_count,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "mean_latency_ms": float(np.mean(inference_times_ms)),
        "Overall": aggregate(all_metrics),
    }

    for c in CVD_TYPES:
        results[c] = aggregate(condition_metrics[c])

    # Save to test_benchmark_results.json
    out_json = backend_root / "test_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal held-out test results saved to: {out_json}")

    # Print Formatted Results
    print("\n" + "=" * 90)
    print("FINAL HELD-OUT TEST SET RESULTS (128 Identities, 384 Condition Pairs)")
    print("=" * 90)
    print(f"{'Condition':<16} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'MS-SSIM':<7} | {'MAE':<7} | {'LPIPS':<7} | {'DeltaE':<7} | {'Count':<5}")
    print("-" * 90)
    ov = results["Overall"]
    print(f"{'OVERALL':<16} | {ov['psnr']:<9.2f} | {ov['ssim']:<7.4f} | {ov['ms_ssim']:<7.4f} | {ov['mae']:<7.4f} | {ov['lpips']:<7.4f} | {ov['ciede2000']:<7.2f} | {ov['count']:<5}")
    print("-" * 90)
    for c in CVD_TYPES:
        cm = results[c]
        print(f"{c:<16} | {cm['psnr']:<9.2f} | {cm['ssim']:<7.4f} | {cm['ms_ssim']:<7.4f} | {cm['mae']:<7.4f} | {cm['lpips']:<7.4f} | {cm['ciede2000']:<7.2f} | {cm['count']:<5}")
    print("=" * 90)
    print(f"Mean CPU Inference Latency: {results['mean_latency_ms']:.2f} ms/image")


if __name__ == "__main__":
    main()

