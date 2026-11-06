# REFERENCE OUTPUTS REPORT: DVAL Restormer_RDB Golden Testing Suite

**Dynamic Visual Accessibility Layer (DVAL)**  
*Personalized Color Vision Deficiency (CVD) Browser Assistant*  
**Date**: November 2026  
**Engine**: `backend/app/core/reference_inference.py`  
**Generator**: `backend/generate_reference_outputs.py`  
**Manifest**: [`backend/reference_outputs/manifest.json`](file:///d:/personalized_colore/backend/reference_outputs/manifest.json)  

---

## 1. Executive Summary

This report establishes the **authoritative static golden reference output suite** for the DVAL browser assistant. The production reference model—`Restormer_RDB` (Epoch 3, 461,694 parameters)—has been executed through the single canonical inference pipeline ([`backend/app/core/reference_inference.py`](file:///d:/personalized_colore/backend/app/core/reference_inference.py)) to produce permanent, bit-exact reference images stored in [`backend/reference_outputs/`](file:///d:/personalized_colore/backend/reference_outputs/).

### Core Achievements
1. **Complete Reference Outputs Generation**:
   - **33 reference images** generated across 11 test subjects (8 representative held-out test split photographs + 3 synthetic calibration charts).
   - **All 3 CVD conditions covered**: Protanopia (Condition ID = 0), Deuteranopia (Condition ID = 1), and Tritanopia (Condition ID = 2).
   - Saved exclusively as **lossless 24-bit RGB PNGs** (`compress_level=6`, zero chroma subsampling, zero lossy quantization).
2. **100% Bit-Exact Determinism & Reproducibility**:
   - Every single generated image was independently verified across duplicate inference passes:
   - $\max | \mathbf{I}_{\text{run1}} - \mathbf{I}_{\text{run2}} | = 0$ (Zero pixel difference).
   - Disk output hashes matched live in-memory execution bit-for-bit.
3. **Structured Reference Architecture**:
   - Standardized layout with `original/`, `protanopia/`, `deuteranopia/`, `tritanopia/`, `comparison/`, `metrics/`, and master `manifest.json`.
   - 4-panel visual comparison strips generated for all subjects: `[ Original | Protanopia (0) | Deuteranopia (1) | Tritanopia (2) ]`.
4. **Automated Master Test Suite Integration**:
   - Tests 28, 29, and 30 added to [`backend/tests/test_master_suite.py`](file:///d:/personalized_colore/backend/tests/test_master_suite.py). All 30/30 unit tests pass in 2.25s.

---

## 2. Production Model Configuration & Cryptographic Identity

All reference outputs are strictly tied to the verified production model:

| Parameter | Authoritative Specification |
|:---|:---|
| **Architecture** | `Restormer_RDB` (Residual Dense Block Conditional Restormer) |
| **Production Checkpoint** | [`backend/weights/best_restormer.pt`](file:///d:/personalized_colore/backend/weights/best_restormer.pt) |
| **Checkpoint SHA-256** | `7f5ad90b372028d17b09fcaaa97b039d824e13d434a09814448af4bca6383eed` |
| **Backup Checkpoint** | [`backend/weights/checkpoints/Restormer_RDB_epoch_003.pt`](file:///d:/personalized_colore/backend/weights/checkpoints/Restormer_RDB_epoch_003.pt) |
| **Model Version** | `1.0.0` |
| **Trainable Parameters** | `461,694` |
| **Training Epoch** | `3` (Validation Loss: `0.009477`) |
| **Residual Formulation** | $\mathbf{R} = 0.5 \times \tanh(\text{OutputProjection}(\mathbf{Z}))$, Output $= \text{clamp}(\mathbf{X} + \mathbf{R}, 0.0, 1.0)$ |
| **Spatial Padding** | Replicate padding to multiples of 8: $H_{\text{pad}} = (8 - H \pmod 8) \pmod 8$, $W_{\text{pad}} = (8 - W \pmod 8) \pmod 8$ |
| **Unpadding** | Exact tensor slice: `[:, :, :orig_h, :orig_w]` |
| **Conditioning** | Spatially broadcasted one-hot 3-channel condition tensor concatenated along dim 1 $\to (B, 6, H_{\text{pad}}, W_{\text{pad}})$ |

---

## 3. Directory Layout & Manifest Specification

All artifacts are organized under [`backend/reference_outputs/`](file:///d:/personalized_colore/backend/reference_outputs/):

```
backend/reference_outputs/
|-- original/                       # Lossless 24-bit RGB PNG source images
|   |-- flower_0001.png
|   |-- flower_0003.png
|   |-- flower_0020.png
|   |-- flower_0034.png
|   |-- flower_0036.png
|   |-- flower_0046.png
|   |-- flower_0066.png
|   |-- flower_0068.png
|   |-- color_chart_macbeth.png
|   |-- color_chart_gradients.png
|   |-- color_chart_cvd_confusion.png
|
|-- protanopia/                     # Corrected reference images (Condition 0)
|   |-- flower_0001_protanopia.png
|   |-- ...
|
|-- deuteranopia/                   # Corrected reference images (Condition 1)
|   |-- flower_0001_deuteranopia.png
|   |-- ...
|
|-- tritanopia/                     # Corrected reference images (Condition 2)
|   |-- flower_0001_tritanopia.png
|   |-- ...
|
|-- comparison/                     # 4-panel visual side-by-side strips [Orig | Pro | Deut | Trit]
|   |-- flower_0001_comparison.png
|   |-- ...
|
|-- metrics/                        # Machine-readable per-image and aggregate JSON files
|   |-- summary_metrics.json
|   |-- flower_0001_metrics.json
|   |-- ...
|
|-- manifest.json                   # Master cryptographic manifest
```

### Manifest Structure (`manifest.json`)
The manifest records:
- `manifest_version`: "1.0.0"
- `model_metadata`: Architecture, checkpoint SHA-256, parameter count, residual formulation, condition mapping.
- `summary`: Total images (11), natural test images (8), synthetic calibration charts (3), total outputs (33), reproducibility verification status (`true`, max diff = 0), and natural test aggregate metrics.
- `images`: Array of comprehensive image entries containing dimensions, file size, SHA-256 hashes for original and each condition, per-condition inference latency, metrics vs. ground truth target, and channel shift residuals.

---

## 4. Source Reference Image Suite

The reference suite consists of two complementary groups:

### 4.1. Natural Held-Out Test Split Photographs
Selected from the locked test split (128 identities, seed 2026), these images span diverse aspect ratios, color palettes, and resolutions:

1. **`flower_0001`** ($190 \times 188$): Prominently featured on [`extension/test_page.html`](file:///d:/personalized_colore/extension/test_page.html). Rich red and pink petals with complex green background foliage.
2. **`flower_0003`** ($112 \times 114$): Compact resolution image testing minimum dimension boundaries and delicate spatial structures.
3. **`flower_0020`** ($273 \times 423$): Tall aspect ratio with mixed warm petals and dark green foliage.
4. **`flower_0034`** ($561 \times 362$): Wide landscape format containing rich yellow, orange, and green tones.
5. **`flower_0036`** ($508 \times 369$): Deep blue/purple floral hues testing Tritanopia correction.
6. **`flower_0046`** ($264 \times 280$): High dynamic range image with white/pink highlights and dark shadows.
7. **`flower_0066`** ($125 \times 189$): Small portrait format with bright green background.
8. **`flower_0068`** ($277 \times 279$): Square composition with high-frequency petal textures.

### 4.2. Controlled Synthetic Calibration Charts
Generated deterministically by [`backend/generate_reference_outputs.py`](file:///d:/personalized_colore/backend/generate_reference_outputs.py) for objective color-space calibration:

1. **`color_chart_macbeth`** ($342 \times 230$): Standard 24-patch Macbeth ColorChecker chart:
   - Row 1: Natural skin, sky, and foliage tones
   - Row 2: Secondary / mixed pigments
   - Row 3: Primary additive and subtractive primaries (R, G, B, C, M, Y)
   - Row 4: 6-step neutral grayscale gradient from white to black
2. **`color_chart_gradients`** ($384 \times 256$): 8 continuous horizontal gradient strips:
   - Red ramp, Green ramp, Blue ramp, Cyan ramp, Magenta ramp, Yellow ramp
   - Full 0–255 neutral grayscale ramp
   - Full $0^\circ \to 360^\circ$ continuous HSV hue spectrum
3. **`color_chart_cvd_confusion`** ($384 \times 256$): 6 diagnostic color confusion pairs designed to verify separation along dichromatic confusion lines:
   - *Protanopia 1*: Red `(220, 30, 30)` vs. Dark Olive Brown `(100, 75, 25)`
   - *Protanopia 2*: Light Green `(115, 205, 80)` vs. Bright Yellow `(220, 215, 35)`
   - *Deuteranopia 1*: Grass Green `(45, 175, 55)` vs. Brick Red `(180, 70, 45)`
   - *Deuteranopia 2*: Teal Cyan `(50, 180, 195)` vs. Neutral Gray `(135, 135, 135)`
   - *Tritanopia 1*: Cobalt Blue `(35, 75, 210)` vs. Forest Green `(30, 135, 60)`
   - *Tritanopia 2*: Canary Yellow `(240, 225, 45)` vs. Rose Pink `(220, 105, 175)`

---

## 5. Quantitative Verification & Metrics Table

### 5.1. Aggregate Natural Test Set Metrics (vs. Ground-Truth Targets)

| CVD Condition | Pairs Evaluated | Mean PSNR (dB) | Mean SSIM | Mean MAE | Mean $\Delta E_{00}$ (CIEDE2000) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall (Natural Suite)** | **24** | **19.9208** | **0.7692** | **0.0753** | **10.6820** |
| **Protanopia (CVD 0)** | 8 | 15.2733 | 0.5804 | 0.1245 | 17.2145 |
| **Deuteranopia (CVD 1)** | 8 | 22.9227 | 0.8627 | 0.0507 | 7.5439 |
| **Tritanopia (CVD 2)** | 8 | 21.5664 | 0.8646 | 0.0506 | 7.2875 |

> **Analysis**: The aggregate metrics across this representative test suite closely align with the comprehensive 384-pair held-out test split evaluated in Phase D (Overall PSNR: 19.69 dB, Deutan: 22.66 dB, Tritan: 20.58 dB, Protan: 15.84 dB). Deuteranopia and Tritanopia achieve high fidelity ($\text{PSNR} > 21.5\text{ dB}$, $\Delta E_{00} \le 7.5$), while Protanopia applies more aggressive compensation ($\text{PSNR} \approx 15.3\text{ dB}$, $\Delta E_{00} \approx 17.2$) to separate red/green boundaries.

---

### 5.2. Detailed Per-Image Metrics Breakdown

#### Protanopia (Condition ID = 0)
| Image ID | Dimensions | PSNR (dB) | SSIM | MAE | $\Delta E_{00}$ vs. Target | $\Delta E_{00}$ vs. Orig | Channel Shift ($\Delta R, \Delta G, \Delta B$) | Output SHA-256 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `flower_0001` | $190 \times 188$ | 17.61 | 0.7394 | 0.0877 | 13.33 | 14.64 | (+0.093, -0.022, +0.019) | `cbcffb82da...` |
| `flower_0003` | $112 \times 114$ | 10.53 | 0.3653 | 0.2223 | 27.91 | 24.16 | (+0.252, -0.076, +0.038) | `538539310e...` |
| `flower_0020` | $273 \times 423$ | 13.20 | 0.4468 | 0.1652 | 20.95 | 22.75 | (+0.144, -0.039, +0.016) | `1de95a1370...` |
| `flower_0034` | $561 \times 362$ | 16.98 | 0.7259 | 0.0911 | 12.75 | 14.07 | (+0.052, -0.013, +0.016) | `f12ac4a718...` |
| `flower_0036` | $508 \times 369$ | 18.81 | 0.7351 | 0.0763 | 8.80 | 12.01 | (+0.081, -0.021, +0.022) | `262e813656...` |
| `flower_0046` | $264 \times 280$ | 18.23 | 0.7291 | 0.0767 | 12.77 | 14.93 | (+0.074, -0.019, +0.018) | `660cd53087...` |
| `flower_0066` | $125 \times 189$ | 12.74 | 0.3956 | 0.1706 | 22.44 | 22.37 | (+0.180, -0.048, +0.018) | `1f0398d8e4...` |
| `flower_0068` | $277 \times 279$ | 14.09 | 0.5061 | 0.1060 | 18.77 | 20.08 | (+0.128, -0.032, +0.017) | `b36d04efe3...` |

#### Deuteranopia (Condition ID = 1)
| Image ID | Dimensions | PSNR (dB) | SSIM | MAE | $\Delta E_{00}$ vs. Target | $\Delta E_{00}$ vs. Orig | Channel Shift ($\Delta R, \Delta G, \Delta B$) | Output SHA-256 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `flower_0001` | $190 \times 188$ | 25.82 | 0.9474 | 0.0352 | 6.39 | 9.62 | (+0.020, +0.029, +0.031) | `fac4e25a19...` |
| `flower_0003` | $112 \times 114$ | 21.33 | 0.8126 | 0.0631 | 11.37 | 12.98 | (+0.037, +0.057, +0.048) | `4904b3bdcd...` |
| `flower_0020` | $273 \times 423$ | 26.08 | 0.9328 | 0.0319 | 5.11 | 9.20 | (+0.021, +0.034, +0.034) | `bf429925b1...` |
| `flower_0034` | $561 \times 362$ | 25.65 | 0.9333 | 0.0328 | 6.28 | 8.89 | (+0.017, +0.027, +0.026) | `f8fa1314bd...` |
| `flower_0036` | $508 \times 369$ | 23.94 | 0.9029 | 0.0443 | 5.37 | 8.05 | (+0.018, +0.028, +0.032) | `5ed4eb6a77...` |
| `flower_0046` | $264 \times 280$ | 23.46 | 0.8988 | 0.0468 | 6.58 | 8.89 | (+0.022, +0.034, +0.033) | `2e46acf8d0...` |
| `flower_0066` | $125 \times 189$ | 16.13 | 0.5360 | 0.1039 | 12.79 | 13.92 | (+0.044, +0.067, +0.042) | `8b80d40264...` |
| `flower_0068` | $277 \times 279$ | 20.97 | 0.8381 | 0.0475 | 6.47 | 10.45 | (+0.025, +0.037, +0.037) | `89c260a56a...` |

#### Tritanopia (Condition ID = 2)
| Image ID | Dimensions | PSNR (dB) | SSIM | MAE | $\Delta E_{00}$ vs. Target | $\Delta E_{00}$ vs. Orig | Channel Shift ($\Delta R, \Delta G, \Delta B$) | Output SHA-256 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `flower_0001` | $190 \times 188$ | 23.68 | 0.9253 | 0.0380 | 6.18 | 8.19 | (+0.019, +0.029, -0.007) | `ddad470834...` |
| `flower_0003` | $112 \times 114$ | 18.10 | 0.7719 | 0.0763 | 11.76 | 13.06 | (+0.035, +0.052, -0.022) | `8c98e2dfba...` |
| `flower_0020` | $273 \times 423$ | 20.24 | 0.8656 | 0.0519 | 8.84 | 9.77 | (+0.021, +0.033, -0.015) | `9f2f34a2a6...` |
| `flower_0034` | $561 \times 362$ | 20.09 | 0.8711 | 0.0515 | 7.71 | 8.65 | (+0.016, +0.026, -0.016) | `d905dec1b4...` |
| `flower_0036` | $508 \times 369$ | 27.63 | 0.9576 | 0.0296 | 3.51 | 6.15 | (+0.016, +0.026, -0.006) | `531ff0f0b7...` |
| `flower_0046` | $264 \times 280$ | 22.46 | 0.9168 | 0.0436 | 6.39 | 7.77 | (+0.020, +0.032, -0.013) | `eb4708e25f...` |
| `flower_0066` | $125 \times 189$ | 21.56 | 0.7423 | 0.0658 | 6.19 | 10.45 | (+0.036, +0.055, -0.027) | `9227628926...` |
| `flower_0068` | $277 \times 279$ | 18.76 | 0.8659 | 0.0481 | 7.72 | 8.58 | (+0.023, +0.034, -0.015) | `0687bd0463...` |

---

### 5.3. Synthetic Calibration Chart Measurements (Behavioral Signatures)

Because calibration charts represent synthetic testing inputs without dataset ground-truth targets, we evaluate their **chromatic shifts and discrimination behavior** relative to the uncorrected chart:

| Synthetic Chart | Condition | Mean Absolute Residual | $\Delta E_{00}$ vs. Original | Channel Shift $(\Delta R, \Delta G, \Delta B)$ | Modified Pixel Ratio | Output SHA-256 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`color_chart_macbeth`** | Protanopia (0) | 0.1345 | 23.91 | (+0.147, -0.038, +0.018) | 99.4% | `bfc476571f...` |
| | Deuteranopia (1) | 0.0988 | 22.16 | (+0.028, +0.045, +0.040) | 99.4% | `31464b76b2...` |
| | Tritanopia (2) | 0.0882 | 19.38 | (+0.027, +0.042, -0.020) | 99.3% | `74722cebd9...` |
| **`color_chart_gradients`** | Protanopia (0) | 0.0880 | 16.90 | (+0.100, -0.024, +0.019) | 99.8% | `6efe697bc6...` |
| | Deuteranopia (1) | 0.0716 | 15.44 | (+0.023, +0.037, +0.035) | 99.8% | `41160ea8a1...` |
| | Tritanopia (2) | 0.0673 | 11.68 | (+0.020, +0.034, -0.015) | 99.8% | `6f11b0d938...` |
| **`color_chart_cvd_confusion`** | Protanopia (0) | 0.0955 | 16.56 | (+0.104, -0.027, +0.020) | 98.7% | `1ae958a358...` |
| | Deuteranopia (1) | 0.0805 | 15.12 | (+0.026, +0.042, +0.037) | 98.7% | `7e4c172574...` |
| | Tritanopia (2) | 0.0746 | 12.82 | (+0.024, +0.038, -0.017) | 98.6% | `52eed0726d...` |

> **Key Behavioral Insights**:
> 1. **Protanopia (Red-blind)**: Restormer consistently applies a substantial positive shift to the Red channel ($\Delta R \approx +0.10 \text{ to } +0.15$) and a negative shift to Green ($\Delta G \approx -0.03$). This differential boosts the luminance and chromatic contrast of red regions relative to adjacent greens and browns.
> 2. **Deuteranopia (Green-blind)**: Restormer elevates Green and Blue ($\Delta G \approx +0.04, \Delta B \approx +0.04$) while maintaining low Red change ($\Delta R \approx +0.02$), effectively remapping greens away from reddish confusability.
> 3. **Tritanopia (Blue-blind)**: Restormer suppresses Blue ($\Delta B \approx -0.017 \text{ to } -0.027$) while elevating Red and Green ($\Delta R \approx +0.02, \Delta G \approx +0.035$), enhancing yellow/blue differentiation.

---

## 6. Reproducibility & Determinism Verification

Reproducibility was evaluated both during reference generation and within the automated master test suite:

### 6.1. In-Script Verification
In [`backend/generate_reference_outputs.py`](file:///d:/personalized_colore/backend/generate_reference_outputs.py):
1. For every image and condition, inference pass 1 and pass 2 were executed sequentially in separate computational graph contexts under `torch.inference_mode()`.
2. $L_\infty$ and $L_1$ differences between the resulting uint8 arrays were calculated:
   $$\text{reproducible\_max\_diff} = \max_{x, y, c} | \mathbf{I}^{(1)}_{x, y, c} - \mathbf{I}^{(2)}_{x, y, c} | = 0$$
3. Result: **33 / 33 outputs achieved 0 pixel difference**.

### 6.2. Master Test Suite Regression Test (Test 29)
In [`backend/tests/test_master_suite.py`](file:///d:/personalized_colore/backend/tests/test_master_suite.py):
- `test_29_golden_reference_bit_exact_reproducibility` loads the saved PNGs from `reference_outputs/` on disk, runs live inference through `AuthoritativeReferenceInference` on the original image, and checks:
  $$\max | \mathbf{I}_{\text{disk}} - \mathbf{I}_{\text{live}} | == 0$$
- Result: **PASSED (0 pixel divergence across all conditions for `flower_0001`)**.

### 6.3. CLI Incremental Reuse
Running:
```bash
python backend/generate_reference_outputs.py --reuse-existing
```
checks existing file hashes against `manifest.json`. If checkpoint SHA-256 and source image hashes match, disk outputs are reused without redundant computation, completing in **0.95 seconds**.

---

## 7. Guidelines on Using Golden References

> [!IMPORTANT]
> **Golden Reference Testing Artifacts vs. Webpage Recoloring**
> - The images in `backend/reference_outputs/` are **golden testing artifacts**.
> - They provide ground-truth expected visual targets against which Chrome Extension canvas rendering, image replacement, and FastAPI network transport can be verified without numerical ambiguity.
> - **They must NOT be used as static Lookup Tables (LUTs)** or universal static color mappings for unrelated webpages.
> - The live Chrome Extension must still pass each page element or image through the verified neural model via the backend API.

---

## 8. Summary of Files Created

1. [`backend/generate_reference_outputs.py`](file:///d:/personalized_colore/backend/generate_reference_outputs.py):
   Authoritative CLI reference output generator supporting `--checkpoint`, `--architecture`, `--conditions`, `--source-images`, `--output-dir`, `--reuse-existing`, and `--verify-reproducibility`.
2. [`backend/reference_outputs/manifest.json`](file:///d:/personalized_colore/backend/reference_outputs/manifest.json):
   Cryptographic master manifest mapping all 33 reference images, dimensions, hashes, and metrics.
3. [`backend/reference_outputs/original/`](file:///d:/personalized_colore/backend/reference_outputs/original/):
   11 lossless 24-bit RGB PNG source images (8 held-out test images + 3 synthetic charts).
4. [`backend/reference_outputs/protanopia/`](file:///d:/personalized_colore/backend/reference_outputs/protanopia/):
   11 corrected reference images for Protanopia (Condition 0).
5. [`backend/reference_outputs/deuteranopia/`](file:///d:/personalized_colore/backend/reference_outputs/deuteranopia/):
   11 corrected reference images for Deuteranopia (Condition 1).
6. [`backend/reference_outputs/tritanopia/`](file:///d:/personalized_colore/backend/reference_outputs/tritanopia/):
   11 corrected reference images for Tritanopia (Condition 2).
7. [`backend/reference_outputs/comparison/`](file:///d:/personalized_colore/backend/reference_outputs/comparison/):
   11 4-panel visual comparison strips with labeled headers (`flower_0001_comparison.png`, etc.).
8. [`backend/reference_outputs/metrics/`](file:///d:/personalized_colore/backend/reference_outputs/metrics/):
   11 per-image JSON metric files and `summary_metrics.json`.
9. [`backend/tests/test_master_suite.py`](file:///d:/personalized_colore/backend/tests/test_master_suite.py):
   Enhanced with Tests 28, 29, and 30 verifying golden manifest integrity, bit-exact reproducibility, and synthetic chart dimensions. (30/30 tests passing).
10. [`d:\personalized_colore\REFERENCE_OUTPUTS_REPORT.md`](file:///d:/personalized_colore/REFERENCE_OUTPUTS_REPORT.md):
   Comprehensive engineering report on the golden reference generation and metrics.

