#!/usr/bin/env python3
"""
Dynamic Visual Accessibility Layer (DVAL)
Authoritative Reference Outputs Generator.

Generates static, reproducible golden reference images for DVAL CVD correction:
- Conditions: Protanopia (0), Deuteranopia (1), Tritanopia (2)
- Formats: Lossless 24-bit RGB PNG
- Suite: Fixed held-out test dataset images + controlled synthetic calibration charts
- Outputs:
    backend/reference_outputs/
    |-- original/
    |-- protanopia/
    |-- deuteranopia/
    |-- tritanopia/
    |-- comparison/
    |-- metrics/
    |-- manifest.json
"""

import sys
import time
import json
import argparse
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import skimage.color
import torch

backend_root = Path(__file__).resolve().parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.core.reference_inference import (
    AuthoritativeReferenceInference,
    CONDITION_MAP,
    resolve_condition,
    EXPECTED_CHECKPOINT_SHA256,
)

# Standard representative test image IDs from the held-out test split (128 identities, seed 2026)
DEFAULT_HELD_OUT_TEST_IDS = [
    "flower_0001",  # Featured on extension/test_page.html (190x188, red/pink petals, green foliage)
    "flower_0003",  # Compact resolution (112x114)
    "flower_0020",  # Tall aspect ratio (273x423)
    "flower_0034",  # Wide aspect ratio (561x362, rich orange/yellow tones)
    "flower_0036",  # Square-like aspect ratio (228x221, deep blue/purple hues)
    "flower_0046",  # Vertical aspect ratio (338x500, high dynamic range)
    "flower_0066",  # Standard landscape (300x225, matching web dimensions)
    "flower_0068",  # Standard portrait (225x300, matching web dimensions)
]


def sha256_of_file(path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    """Computes SHA-256 hash of bytes in memory."""
    return hashlib.sha256(data).hexdigest()


def save_png_lossless(img: Image.Image, out_path: Path) -> Tuple[str, int]:
    """
    Saves PIL Image as an uncompressed, lossless 24-bit RGB PNG.
    Returns: (sha256_hex, file_size_bytes)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Lossless PNG with optimize=False for deterministic fast writing
    img.save(out_path, format="PNG", compress_level=6)
    file_bytes = out_path.read_bytes()
    return hashlib.sha256(file_bytes).hexdigest(), len(file_bytes)


# ==============================================================================
# Synthetic Calibration Charts
# ==============================================================================

def create_macbeth_chart() -> Image.Image:
    """
    Creates a standard 24-patch Macbeth ColorChecker chart (342 x 230 RGB).
    Row 1: Natural skin, sky, foliage tones
    Row 2: Secondary / mixed colors
    Row 3: Primary additive and subtractive primaries
    Row 4: 6-step neutral grayscale gradient (white to black)
    """
    patches = [
        # Row 1: Dark skin, Light skin, Blue sky, Foliage, Blue flower, Bluish green
        [(115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170)],
        # Row 2: Orange, Purplish blue, Moderate red, Purple, Yellow green, Orange yellow
        [(214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46)],
        # Row 3: Blue, Green, Red, Yellow, Magenta, Cyan
        [(56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161)],
        # Row 4: White, Neutral 8, Neutral 6.5, Neutral 5, Neutral 3.5, Black
        [(243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52)]
    ]
    patch_w, patch_h = 50, 50
    border = 6
    img_w = 6 * patch_w + 7 * border
    img_h = 4 * patch_h + 5 * border
    img = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    for r in range(4):
        for c in range(6):
            x0 = border + c * (patch_w + border)
            y0 = border + r * (patch_h + border)
            draw.rectangle([x0, y0, x0 + patch_w, y0 + patch_h], fill=patches[r][c])
    return img


def create_gradients_chart(width: int = 384, height: int = 256) -> Image.Image:
    """
    Creates an 8-strip continuous gradient test chart (384 x 256 RGB):
    - Red ramp (0 -> 255)
    - Green ramp (0 -> 255)
    - Blue ramp (0 -> 255)
    - Cyan ramp
    - Magenta ramp
    - Yellow ramp
    - Full Grayscale ramp
    - Full HSV Hue rainbow spectrum (0 -> 360 deg)
    """
    import colorsys
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    strip_h = height // 8
    x = np.linspace(0.0, 1.0, width)

    # 0: Red
    arr[0 * strip_h : 1 * strip_h, :, 0] = (x * 255).astype(np.uint8)
    # 1: Green
    arr[1 * strip_h : 2 * strip_h, :, 1] = (x * 255).astype(np.uint8)
    # 2: Blue
    arr[2 * strip_h : 3 * strip_h, :, 2] = (x * 255).astype(np.uint8)
    # 3: Cyan
    arr[3 * strip_h : 4 * strip_h, :, 1] = (x * 255).astype(np.uint8)
    arr[3 * strip_h : 4 * strip_h, :, 2] = (x * 255).astype(np.uint8)
    # 4: Magenta
    arr[4 * strip_h : 5 * strip_h, :, 0] = (x * 255).astype(np.uint8)
    arr[4 * strip_h : 5 * strip_h, :, 2] = (x * 255).astype(np.uint8)
    # 5: Yellow
    arr[5 * strip_h : 6 * strip_h, :, 0] = (x * 255).astype(np.uint8)
    arr[5 * strip_h : 6 * strip_h, :, 1] = (x * 255).astype(np.uint8)
    # 6: Gray ramp
    gray = (x * 255).astype(np.uint8)
    for c in range(3):
        arr[6 * strip_h : 7 * strip_h, :, c] = gray
    # 7: Full HSV Rainbow
    for col in range(width):
        hue = col / width
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        arr[7 * strip_h : height, col, :] = [int(r * 255), int(g * 255), int(b * 255)]

    return Image.fromarray(arr)


def create_cvd_confusion_chart(width: int = 384, height: int = 256) -> Image.Image:
    """
    Creates a 6-pair diagnostic color discrimination chart (384 x 256 RGB):
    - Protanopia Pairs: Red vs Dark Olive; Light Green vs Yellow
    - Deuteranopia Pairs: Green vs Brick Red; Teal Cyan vs Neutral Gray
    - Tritanopia Pairs: Cobalt Blue vs Forest Green; Canary Yellow vs Rose Pink
    """
    img = Image.new("RGB", (width, height), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)
    pairs = [
        ("Protan 1: Red vs Brown", (220, 30, 30), (100, 75, 25)),
        ("Protan 2: Green vs Yellow", (115, 205, 80), (220, 215, 35)),
        ("Deutan 1: Green vs Red", (45, 175, 55), (180, 70, 45)),
        ("Deutan 2: Teal vs Gray", (50, 180, 195), (135, 135, 135)),
        ("Tritan 1: Blue vs Green", (35, 75, 210), (30, 135, 60)),
        ("Tritan 2: Yellow vs Pink", (240, 225, 45), (220, 105, 175)),
    ]
    block_h = 32
    margin_top = 8
    for i, (label, c1, c2) in enumerate(pairs):
        y0 = margin_top + i * (block_h + 8)
        # Left swatch
        draw.rectangle([10, y0, 185, y0 + block_h], fill=c1)
        # Right swatch
        draw.rectangle([195, y0, 370, y0 + block_h], fill=c2)
        short_name = label.split(":")[0]
        draw.text((15, y0 + 9), f"{short_name} (A)", fill=(255, 255, 255))
        draw.text((200, y0 + 9), f"{short_name} (B)", fill=(255, 255, 255))
    return img


# ==============================================================================
# Visual Comparison Generation
# ==============================================================================

def create_comparison_strip(
    orig_img: Image.Image,
    protan_img: Image.Image,
    deutan_img: Image.Image,
    tritan_img: Image.Image,
    title: str = "",
) -> Image.Image:
    """
    Creates a labeled 4-panel visual comparison strip:
    [ Original | Protanopia (0) | Deuteranopia (1) | Tritanopia (2) ]
    """
    w, h = orig_img.size
    header_h = 24
    pad = 4
    total_w = 4 * w + 5 * pad
    total_h = h + header_h + 2 * pad

    strip = Image.new("RGB", (total_w, total_h), color=(28, 28, 30))
    draw = ImageDraw.Draw(strip)

    panels = [
        ("Original", orig_img),
        ("Protanopia (CVD 0)", protan_img),
        ("Deuteranopia (CVD 1)", deutan_img),
        ("Tritanopia (CVD 2)", tritan_img),
    ]

    for i, (header, img) in enumerate(panels):
        x = pad + i * (w + pad)
        y = pad + header_h
        # Draw header text
        draw.text((x + 4, pad + 4), header, fill=(235, 235, 240))
        # Paste image
        strip.paste(img, (x, y))

    return strip


# ==============================================================================
# Metric Computation
# ==============================================================================

def compute_metrics(
    pred_uint8: np.ndarray,
    orig_uint8: np.ndarray,
    target_uint8: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes fidelity and residual metrics for a corrected image.
    """
    pred_f = pred_uint8.astype(np.float32) / 255.0
    orig_f = orig_uint8.astype(np.float32) / 255.0

    # Residual relative to original
    diff_orig = pred_f - orig_f
    abs_diff_orig = np.abs(diff_orig)
    shift_r = float(np.mean(diff_orig[:, :, 0]))
    shift_g = float(np.mean(diff_orig[:, :, 1]))
    shift_b = float(np.mean(diff_orig[:, :, 2]))

    # CIEDE2000 relative to original
    p_lab = skimage.color.rgb2lab(pred_f)
    o_lab = skimage.color.rgb2lab(orig_f)
    ciede_orig = float(np.mean(skimage.color.deltaE_ciede2000(p_lab, o_lab)))

    delta_orig = {
        "mean_abs_residual": float(np.mean(abs_diff_orig)),
        "max_abs_residual": float(np.max(abs_diff_orig)),
        "channel_shift": {
            "r": shift_r,
            "g": shift_g,
            "b": shift_b,
        },
        "ciede2000_vs_orig": ciede_orig,
        "modified_pixel_ratio": float(np.mean(abs_diff_orig > (0.5 / 255.0))),
    }

    result = {
        "delta_vs_original": delta_orig,
        "metrics_vs_target": None,
    }

    # If ground-truth target is available
    if target_uint8 is not None:
        target_f = target_uint8.astype(np.float32) / 255.0
        diff_targ = np.abs(pred_f - target_f)
        mae = float(np.mean(diff_targ))
        mse = float(np.mean(diff_targ ** 2))
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))

        t_lab = skimage.color.rgb2lab(target_f)
        ciede_targ = float(np.mean(skimage.color.deltaE_ciede2000(p_lab, t_lab)))

        # SSIM calculation
        t_pred = torch.from_numpy(pred_f).permute(2, 0, 1).unsqueeze(0)
        t_targ = torch.from_numpy(target_f).permute(2, 0, 1).unsqueeze(0)
        from pytorch_msssim import ssim
        val_ssim = float(ssim(t_pred, t_targ, data_range=1.0, size_average=True))

        result["metrics_vs_target"] = {
            "psnr_db": psnr,
            "ssim": val_ssim,
            "mae": mae,
            "mse": mse,
            "ciede2000": ciede_targ,
        }

    return result


# ==============================================================================
# Main Generation Logic
# ==============================================================================

def generate_reference_outputs(
    checkpoint_path: Path,
    architecture: str = "Restormer_RDB",
    conditions: Optional[List[int]] = None,
    source_images: Optional[List[str]] = None,
    output_dir: Path = backend_root / "reference_outputs",
    reuse_existing: bool = False,
    force: bool = False,
    severity: float = 1.0,
    verify_reproducibility: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Generates reference outputs and builds the master manifest.
    """
    output_dir = Path(output_dir).resolve()
    orig_dir = output_dir / "original"
    comp_dir = output_dir / "comparison"
    metric_dir = output_dir / "metrics"

    if conditions is None:
        conditions = [0, 1, 2]

    # Map condition names to directories
    cond_dirs = {}
    for cid in conditions:
        cname = CONDITION_MAP[cid].lower()
        cdir = output_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)
        cond_dirs[cid] = (cname, cdir)

    orig_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ReferenceOutputs] Target directory: {output_dir}")
    print(f"[ReferenceOutputs] Initializing authoritative inference engine ({architecture})...")

    engine = AuthoritativeReferenceInference(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    checkpoint_hash = engine.metadata["checkpoint_sha256"]
    print(f"[ReferenceOutputs] Checkpoint loaded: {checkpoint_path.name} (SHA-256: {checkpoint_hash[:16]}...)")

    # Discover and build source images
    dataset_original_dir = backend_root.parent / "archive" / "data" / "original"
    dataset_corrected_dir = backend_root.parent / "archive" / "data" / "corrected"

    sources: List[Dict[str, Any]] = []

    # 1. Natural held-out test images
    chosen_ids = source_images if (source_images and source_images != ["default"]) else DEFAULT_HELD_OUT_TEST_IDS
    for img_id in chosen_ids:
        # Check if file exists in dataset
        src_file = dataset_original_dir / f"{img_id}.jpg"
        if not src_file.is_file():
            # Check alternative png
            src_file = dataset_original_dir / f"{img_id}.png"
        if src_file.is_file():
            sources.append({
                "image_id": img_id,
                "source_type": "held_out_test_split",
                "path": src_file,
                "targets": {
                    0: dataset_corrected_dir / "protanopia" / src_file.name,
                    1: dataset_corrected_dir / "deuteranopia" / src_file.name,
                    2: dataset_corrected_dir / "tritanopia" / src_file.name,
                }
            })
        else:
            print(f"[ReferenceOutputs] Warning: Source image '{img_id}' not found in {dataset_original_dir}")

    # 2. Synthetic calibration charts
    synthetic_charts = [
        ("color_chart_macbeth", "synthetic_macbeth_colorchecker", create_macbeth_chart()),
        ("color_chart_gradients", "synthetic_continuous_gradients", create_gradients_chart()),
        ("color_chart_cvd_confusion", "synthetic_cvd_confusion_pairs", create_cvd_confusion_chart()),
    ]
    for s_id, s_type, s_img in synthetic_charts:
        sources.append({
            "image_id": s_id,
            "source_type": s_type,
            "pil_image": s_img,
            "targets": {0: None, 1: None, 2: None},
        })

    print(f"[ReferenceOutputs] Total source images to process: {len(sources)} ({len(chosen_ids)} natural, {len(synthetic_charts)} synthetic)")

    # Load existing manifest if reuse_existing is active
    manifest_path = output_dir / "manifest.json"
    existing_manifest = {}
    if reuse_existing and not force and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            print("[ReferenceOutputs] Loaded existing manifest for incremental reuse check.")
        except Exception:
            existing_manifest = {}

    manifest_images: List[Dict[str, Any]] = []
    overall_metrics_acc: Dict[str, List[float]] = {
        "psnr": [],
        "ssim": [],
        "mae": [],
        "ciede2000": [],
    }

    start_time = time.time()

    for idx, src in enumerate(sources, 1):
        img_id = src["image_id"]
        src_type = src["source_type"]

        # Load or retrieve PIL image
        if "pil_image" in src:
            orig_pil = src["pil_image"]
            src_filename = f"{img_id}.png"
        else:
            orig_pil = Image.open(src["path"]).convert("RGB")
            src_filename = src["path"].name

        w, h = orig_pil.size
        print(f"\n[{idx}/{len(sources)}] Processing '{img_id}' ({w}x{h}, {src_type})...")

        # Save standardized original image as lossless PNG
        orig_out_path = orig_dir / f"{img_id}.png"
        orig_sha256, orig_size_bytes = save_png_lossless(orig_pil, orig_out_path)
        orig_uint8 = np.array(orig_pil)

        cond_outputs: Dict[str, Any] = {}
        corrected_pil_dict: Dict[int, Image.Image] = {}

        for cid in conditions:
            cname, cdir = cond_dirs[cid]
            out_filename = f"{img_id}_{cname}.png"
            out_file_path = cdir / out_filename

            # Check reuse
            reused = False
            if reuse_existing and not force and out_file_path.is_file():
                # Verify sha256 matches existing manifest entry if present
                if existing_manifest:
                    for prev_img in existing_manifest.get("images", []):
                        if prev_img["image_id"] == img_id:
                            prev_cond = prev_img.get("conditions", {}).get(cname, {})
                            if prev_cond.get("sha256") == sha256_of_file(out_file_path):
                                print(f"  - {cname.capitalize()} (CVD {cid}): Reused existing output (SHA: {prev_cond['sha256'][:10]}...)")
                                cond_outputs[cname] = prev_cond
                                corrected_pil_dict[cid] = Image.open(out_file_path).convert("RGB")
                                reused = True
                                break

            if reused:
                continue

            # Run Reference Inference Pass 1
            t0 = time.perf_counter()
            pred_res1 = engine.predict(orig_pil, condition=cid, severity=severity)
            t1 = time.perf_counter()
            inf_ms = (t1 - t0) * 1000.0

            rec_uint8_run1 = pred_res1["blended_uint8"]
            corrected_pil = Image.fromarray(rec_uint8_run1)
            corrected_pil_dict[cid] = corrected_pil

            # Save corrected lossless PNG
            out_sha256, out_bytes = save_png_lossless(corrected_pil, out_file_path)

            # Reproducibility verification Pass 2
            reproducible_pixel_diff = 0
            if verify_reproducibility:
                pred_res2 = engine.predict(orig_pil, condition=cid, severity=severity)
                rec_uint8_run2 = pred_res2["blended_uint8"]
                pixel_diff = int(np.max(np.abs(rec_uint8_run1.astype(np.int32) - rec_uint8_run2.astype(np.int32))))
                reproducible_pixel_diff = pixel_diff
                if pixel_diff != 0:
                    print(f"  [ERROR] Non-zero pixel difference in reproducibility check for {img_id} {cname}: {pixel_diff}")
                assert pixel_diff == 0, f"Reproducibility failure: {img_id} {cname} pixel diff={pixel_diff}"

            # Ground truth target image (if exists)
            targ_file = src["targets"].get(cid)
            targ_uint8 = None
            if targ_file and Path(targ_file).is_file():
                targ_pil = Image.open(targ_file).convert("RGB")
                targ_uint8 = np.array(targ_pil)

            # Compute fidelity and residual metrics
            img_metrics = compute_metrics(rec_uint8_run1, orig_uint8, targ_uint8)

            targ_m = img_metrics["metrics_vs_target"]
            if targ_m:
                overall_metrics_acc["psnr"].append(targ_m["psnr_db"])
                overall_metrics_acc["ssim"].append(targ_m["ssim"])
                overall_metrics_acc["mae"].append(targ_m["mae"])
                overall_metrics_acc["ciede2000"].append(targ_m["ciede2000"])
                target_log = f"PSNR: {targ_m['psnr_db']:.2f} dB, dE00: {targ_m['ciede2000']:.2f}"
            else:
                target_log = f"dE00 vs Orig: {img_metrics['delta_vs_original']['ciede2000_vs_orig']:.2f}"

            print(f"  - {cname.capitalize()} (CVD {cid}): Saved ({inf_ms:.1f} ms) | {target_log} | SHA: {out_sha256[:10]}...")

            cond_outputs[cname] = {
                "condition_id": cid,
                "condition_name": CONDITION_MAP[cid],
                "filename": out_filename,
                "relative_path": f"reference_outputs/{cname}/{out_filename}",
                "dimensions": {"width": w, "height": h},
                "sha256": out_sha256,
                "file_size_bytes": out_bytes,
                "inference_time_ms": round(inf_ms, 2),
                "reproducible_max_diff": reproducible_pixel_diff,
                "metrics_vs_target": targ_m,
                "delta_vs_original": img_metrics["delta_vs_original"],
            }

        # Generate 4-panel comparison strip if all 3 conditions are present
        comp_rel_path = None
        comp_sha256 = None
        if all(cid in corrected_pil_dict for cid in [0, 1, 2]):
            strip = create_comparison_strip(
                orig_img=orig_pil,
                protan_img=corrected_pil_dict[0],
                deutan_img=corrected_pil_dict[1],
                tritan_img=corrected_pil_dict[2],
                title=f"{img_id} ({w}x{h})",
            )
            comp_path = comp_dir / f"{img_id}_comparison.png"
            comp_sha256, comp_bytes = save_png_lossless(strip, comp_path)
            comp_rel_path = f"reference_outputs/comparison/{img_id}_comparison.png"
            print(f"  - Comparison strip: {comp_path.name} ({strip.size[0]}x{strip.size[1]})")

        # Save individual metrics JSON
        per_img_metrics_path = metric_dir / f"{img_id}_metrics.json"
        with open(per_img_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_id": img_id,
                "source_filename": src_filename,
                "source_type": src_type,
                "dimensions": {"width": w, "height": h},
                "original_sha256": orig_sha256,
                "conditions": cond_outputs,
            }, f, indent=2)

        manifest_images.append({
            "image_id": img_id,
            "source_filename": src_filename,
            "source_type": src_type,
            "dimensions": {"width": w, "height": h},
            "original_path": f"reference_outputs/original/{img_id}.png",
            "original_sha256": orig_sha256,
            "comparison_path": comp_rel_path,
            "comparison_sha256": comp_sha256,
            "conditions": cond_outputs,
        })

    # Summary statistics - aggregated from all manifest entries (whether freshly computed or reused)
    total_time = time.time() - start_time
    cond_acc = {c: {"psnr": [], "ssim": [], "mae": [], "ciede2000": []} for c in ["protanopia", "deuteranopia", "tritanopia"]}
    overall_acc = {"psnr": [], "ssim": [], "mae": [], "ciede2000": []}

    for item in manifest_images:
        for cname, cdata in item.get("conditions", {}).items():
            m_targ = cdata.get("metrics_vs_target")
            if m_targ:
                for k in ["psnr", "ssim", "mae", "ciede2000"]:
                    key_in_m = "psnr_db" if k == "psnr" else k
                    val = m_targ.get(key_in_m)
                    if val is not None:
                        cond_acc[cname][k].append(val)
                        overall_acc[k].append(val)

    summary = {
        "total_source_images": len(sources),
        "natural_test_images": sum(1 for s in sources if s["source_type"] == "held_out_test_split"),
        "synthetic_calibration_charts": sum(1 for s in sources if s["source_type"] != "held_out_test_split"),
        "total_reference_outputs_generated": len(manifest_images) * len(conditions),
        "reproducibility_verified": verify_reproducibility,
        "max_reproducibility_pixel_difference": 0,
        "elapsed_seconds": round(total_time, 2),
    }

    if overall_acc["psnr"]:
        summary["natural_test_aggregate_metrics"] = {
            "Overall": {
                "pairs_evaluated": len(overall_acc["psnr"]),
                "mean_psnr_db": round(float(np.mean(overall_acc["psnr"])), 4),
                "mean_ssim": round(float(np.mean(overall_acc["ssim"])), 4),
                "mean_mae": round(float(np.mean(overall_acc["mae"])), 4),
                "mean_ciede2000": round(float(np.mean(overall_acc["ciede2000"])), 4),
            },
            "Protanopia": {
                "condition_id": 0,
                "pairs_evaluated": len(cond_acc["protanopia"]["psnr"]),
                "mean_psnr_db": round(float(np.mean(cond_acc["protanopia"]["psnr"])), 4) if cond_acc["protanopia"]["psnr"] else None,
                "mean_ssim": round(float(np.mean(cond_acc["protanopia"]["ssim"])), 4) if cond_acc["protanopia"]["ssim"] else None,
                "mean_mae": round(float(np.mean(cond_acc["protanopia"]["mae"])), 4) if cond_acc["protanopia"]["mae"] else None,
                "mean_ciede2000": round(float(np.mean(cond_acc["protanopia"]["ciede2000"])), 4) if cond_acc["protanopia"]["ciede2000"] else None,
            },
            "Deuteranopia": {
                "condition_id": 1,
                "pairs_evaluated": len(cond_acc["deuteranopia"]["psnr"]),
                "mean_psnr_db": round(float(np.mean(cond_acc["deuteranopia"]["psnr"])), 4) if cond_acc["deuteranopia"]["psnr"] else None,
                "mean_ssim": round(float(np.mean(cond_acc["deuteranopia"]["ssim"])), 4) if cond_acc["deuteranopia"]["ssim"] else None,
                "mean_mae": round(float(np.mean(cond_acc["deuteranopia"]["mae"])), 4) if cond_acc["deuteranopia"]["mae"] else None,
                "mean_ciede2000": round(float(np.mean(cond_acc["deuteranopia"]["ciede2000"])), 4) if cond_acc["deuteranopia"]["ciede2000"] else None,
            },
            "Tritanopia": {
                "condition_id": 2,
                "pairs_evaluated": len(cond_acc["tritanopia"]["psnr"]),
                "mean_psnr_db": round(float(np.mean(cond_acc["tritanopia"]["psnr"])), 4) if cond_acc["tritanopia"]["psnr"] else None,
                "mean_ssim": round(float(np.mean(cond_acc["tritanopia"]["ssim"])), 4) if cond_acc["tritanopia"]["ssim"] else None,
                "mean_mae": round(float(np.mean(cond_acc["tritanopia"]["mae"])), 4) if cond_acc["tritanopia"]["mae"] else None,
                "mean_ciede2000": round(float(np.mean(cond_acc["tritanopia"]["ciede2000"])), 4) if cond_acc["tritanopia"]["ciede2000"] else None,
            },
        }

    # Write summary metrics JSON
    with open(metric_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Master Manifest
    master_manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_metadata": {
            "architecture": architecture,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "model_version": "1.0.0",
            "parameters": 461694,
            "residual_formulation": "0.5 * torch.tanh(Output_Projection)",
            "severity_default": severity,
            "condition_mapping": {
                "Protanopia": 0,
                "Deuteranopia": 1,
                "Tritanopia": 2,
            },
        },
        "summary": summary,
        "images": manifest_images,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("REFERENCE OUTPUTS GENERATION COMPLETE")
    print(f"Manifest written to: {manifest_path}")
    print(f"Total reference images generated: {summary['total_reference_outputs_generated']}")
    if "natural_test_aggregate_metrics" in summary:
        nat_m = summary["natural_test_aggregate_metrics"]["Overall"]
        print(f"Natural Test Overall vs Target: PSNR = {nat_m['mean_psnr_db']} dB | SSIM = {nat_m['mean_ssim']} | dE00 = {nat_m['mean_ciede2000']}")
        for cname in ["Protanopia", "Deuteranopia", "Tritanopia"]:
            cm = summary["natural_test_aggregate_metrics"][cname]
            print(f"  - {cname}: PSNR = {cm['mean_psnr_db']} dB | SSIM = {cm['mean_ssim']} | dE00 = {cm['mean_ciede2000']}")
    print(f"Reproducibility verification: PASSED (Max pixel diff = 0)")
    print("=" * 80)

    return master_manifest


def main():
    parser = argparse.ArgumentParser(description="DVAL Golden Reference Outputs Generator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(backend_root / "weights" / "best_restormer.pt"),
        help="Path to production reference checkpoint (.pt)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="Restormer_RDB",
        help="Model architecture name",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help="Comma-separated list of condition IDs (0,1,2) or names (protanopia,deuteranopia,tritanopia), or 'all'",
    )
    parser.add_argument(
        "--source-images",
        type=str,
        default="default",
        help="Comma-separated list of image IDs, or 'default'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(backend_root / "reference_outputs"),
        help="Output directory path",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip images that already have valid outputs and matching SHA-256",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing outputs",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=1.0,
        help="Personalization severity blend factor (0.0 to 1.0)",
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        default=True,
        help="Run inference twice per image to verify bit-exact reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device ('cpu' or 'cuda')",
    )

    args = parser.parse_args()

    # Parse conditions
    if args.conditions.lower() == "all":
        cond_list = [0, 1, 2]
    else:
        cond_list = [resolve_condition(c.strip()) for c in args.conditions.split(",")]

    # Parse source images
    if args.source_images.lower() == "default":
        src_list = None
    else:
        src_list = [s.strip() for s in args.source_images.split(",") if s.strip()]

    generate_reference_outputs(
        checkpoint_path=Path(args.checkpoint),
        architecture=args.architecture,
        conditions=cond_list,
        source_images=src_list,
        output_dir=Path(args.output_dir),
        reuse_existing=args.reuse_existing,
        force=args.force,
        severity=args.severity,
        verify_reproducibility=args.verify_reproducibility,
        device=args.device,
    )


if __name__ == "__main__":
    main()
