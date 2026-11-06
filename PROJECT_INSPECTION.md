# Dynamic Visual Accessibility Layer (DVAL)
# Complete Project Inspection Report (Phase A)

**Date**: 2026-11-04  
**Engineer**: Senior Computer Vision & Systems Engineer  
**Workspace**: `d:\personalized_colore`  
**System Environment**: Windows 11, Intel Core i5-1155G7 (4 physical cores, 8 logical processors), ~7.8 GiB RAM, PyTorch 2.14.0+cpu, Python 3.11.15.

---

## 1. Executive Summary & Objective

This inspection provides a thorough, rigorous audit of the **Dynamic Visual Accessibility Layer (DVAL)** codebase for Personalized Color Vision Deficiency (CVD) assistance.

The primary objective of this project is:
> **The color correction produced by the trained PyTorch neural model must be faithfully and consistently reproduced by the existing Chrome extension in the browser.** The model is the ground truth; the backend must preserve its exact numerical color space; and the extension must display that correction without unintended color shifts, lossy compression artifacts, or affine approximations.

Before executing any modifications, every component of the repository was inspected: model architectures, checkpoint state dictionaries, preprocessing and postprocessing routines, dataset configuration and pairing, FastAPI routes, caching systems, extension manifest, background service workers, content scripts, popup UI, test scripts, and historical audit logs.

---

## 2. Directory Structure & File Inventory

The actual project structure matches the expected architecture with minor platform-specific attributes:

```
d:\personalized_colore/
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   |-- __init__.py
|   |   |   |-- routes.py              (FastAPI endpoints: recolor, health, plates, diagnose, filter)
|   |   |   |-- schemas.py             (Pydantic input/output schemas)
|   |   |-- core/
|   |   |   |-- __init__.py
|   |   |   |-- cache.py               (Thread-safe bounded LRU cache with SHA256 key)
|   |   |   |-- model.py               (6 model architectures: Restormer variants, UNet, CNN)
|   |   |   |-- page_filter.py         (Affine least-squares 4x5 SVG feColorMatrix fitting)
|   |   |   |-- personalization.py     (Condition ID resolution & severity alpha blending)
|   |   |   |-- postprocessing.py      (Unpadding, [0,1] clamping, uint8, WebP/JPEG/PNG encoding)
|   |   |   |-- predictor.py           (Singleton predictor manager, torch.inference_mode)
|   |   |   |-- preprocessing.py       (Base64 decode, EXIF transpose, RGB, pad to mod 8)
|   |   |   |-- screening.py           (Non-clinical screening scoring engine)
|   |   |-- config.py                  (Environment settings, default weights path)
|   |   |-- main.py                    (FastAPI app factory, CORS middleware)
|   |-- screening/
|   |   |-- plates.json                (5 calibrated CIE L*a*b* screening plate definitions)
|   |   |-- README.md                  (Screening documentation & clinical disclaimer)
|   |-- tests/
|   |   |-- test_backend.py            (Endpoint & plate integration tests)
|   |   |-- test_master_suite.py       (24 master test cases: models, cache, formats, leakage)
|   |-- weights/
|   |   |-- checkpoints/
|   |   |   |-- Restormer_RDB_epoch_001.pt (Epoch 1 checkpoint, 5.63 MB)
|   |   |   |-- Restormer_RDB_epoch_002.pt (Epoch 2 checkpoint, 5.63 MB)
|   |   |   |-- Restormer_RDB_epoch_003.pt (Epoch 3 checkpoint, 5.63 MB)
|   |   |-- best_restormer.pt          (Selected production weights, Epoch 3, 5.63 MB)
|   |   |-- last_checkpoint.pt         (Epoch 3 resume checkpoint, 5.63 MB)
|   |   |-- README.md                  (Weights directory documentation)
|   |-- requirements.txt               (Python dependencies: torch, torchvision, fastapi, etc.)
|   |-- run_server.py                  (Uvicorn entrypoint: host 0.0.0.0, port 8000)
|   |-- train_or_eval.py               (PyTorch training & evaluation pipeline)
|   |-- README.md                      (Backend documentation)
|
|-- extension/
|   |-- icons/                         (icon-16.png, icon-48.png, icon-128.png)
|   |-- popup/
|   |   |-- plates_data.js             (Client-side CIE L*a*b* dot-mosaic Ishihara generator)
|   |   |-- popup.css                  (Extension styling, tabs, responsive test-wide mode)
|   |   |-- popup.html                 (Dual-tab interface: Personalization & Ishihara Test)
|   |   |-- popup.js                   (Popup interaction, storage sync, health indicator)
|   |-- scripts/
|   |   |-- background.js              (MV3 service worker: cross-origin fetch, proxying)
|   |   |-- content.js                 (DOM scanner, MutationObserver, SVG filter & per-img recolor)
|   |-- manifest.json                  (Manifest V3 configuration & permissions)
|   |-- test_page.html                 (Interactive integration test page with flower images)
|
|-- archive/
|   |-- data/
|       |-- corrected/
|       |   |-- protanopia/            (858 ground truth corrected targets)
|       |   |-- deuteranopia/          (858 ground truth corrected targets)
|       |   |-- tritanopia/            (858 ground truth corrected targets)
|       |-- original/                  (858 ground truth original images)
|
|-- dataset                            (Text file pointer containing 'archive/data')
|-- audit.txt                          (Comprehensive historical technical audit log)
|-- README.md                          (Root repository documentation)
|-- validation-first-conditional-restormer-six-seed-a.ipynb (Research & ablation notebook)
```

---

## 3. Detailed Component Inspection

### 3.1 Model Architectures & Factory (`backend/app/core/model.py`)
- **Conditioning Representation**: Input image $\mathbf{X} \in \mathbb{R}^{B \times 3 \times H \times W}$ is normalized to $[0.0, 1.0]$. The scalar `condition_id` $\in \{0, 1, 2\}$ is transformed via `F.one_hot(condition_id, num_classes=3)` into a 3-channel spatial map $(B, 3, H, W)$ and concatenated along the channel dimension, yielding a 6-channel tensor $(B, 6, H, W)$ for `patch_embedding`.
- **Architectures Supported**:
  1. `Restormer_RDB`: 461,694 parameters (70 parameter tensors). Uses Multi-Dconv Head Transposed Attention (MDTA), Gated Depthwise Feed-Forward Networks (GDFN), and a Residual Dense Block (RDB) in the latent bottleneck (3 dense $3 \times 3$ convs with GELU + $1 \times 1$ conv fusion).
  2. `Modified_Conditional_Restormer`: 462,888 parameters (76 parameter tensors). Identical to `Restormer_RDB` but includes CBAM (Convolutional Block Attention Module: Channel + $7 \times 7$ Spatial Attention) after the RDB.
  3. `Restormer_CBAM`: 225,064 parameters. Includes CBAM without RDB.
  4. `Base_Restormer_Style`: 223,870 parameters. Pure Restormer hierarchy without RDB or CBAM.
  5. `Conditional_Residual_UNet`: 1,927,939 parameters. 4-level encoder-decoder baseline with ConvTranspose2d upsampling and skip connections.
  6. `Plain_Conditional_Residual_CNN`: 337,251 parameters. 8 residual CNN blocks with GroupNorm and GELU.
- **Residual Formulation**:
  All models strictly follow the bounded residual formulation:
  $$\Delta = 0.5 \times \tanh(\text{OutputProjection}(\mathbf{Z}))$$
  $$\mathbf{Y} = \text{clamp}(\mathbf{X} + \Delta, 0.0, 1.0)$$
  This guarantees that predictions never diverge beyond $\pm 0.5$ additive color shifts from the input image and remain strictly within $[0.0, 1.0]$.

### 3.2 Preprocessing & Postprocessing Pipelines
- **Preprocessing (`backend/app/core/preprocessing.py`)**:
  - `decode_base64_image`: Validates Base64 string, strips data URL scheme prefix (`data:image/...;base64,`), verifies size limit ($\le 10$ MB), loads PIL Image, executes `ImageOps.exif_transpose` (preventing rotation flips on mobile uploads), ensures `RGB` mode, and safely downsamples if $\max(W, H) > 2048$ with bicubic interpolation.
  - `prepare_input_tensor`: Normalizes uint8 $[0, 255] \to \text{float32} [0.0, 1.0]$, converts $(H, W, 3) \to (1, 3, H, W)$, and pads spatial dimensions $(H, W)$ to multiples of 8 using `mode="replicate"`.
- **Postprocessing (`backend/app/core/postprocessing.py`)**:
  - `unpad_and_clamp_tensor`: Crops the output tensor back to original $(H, W)$, detaches, clamps to $[0.0, 1.0]$, multiplies by 255.0, applies `.round()`, and converts to `uint8` $(H, W, 3)$.
  - `encode_base64_image`: Encodes the `uint8` NumPy array using PIL into Base64 data URL. Supports `webp` (quality 90, method 4), `jpeg` (quality 90), and `png` (`optimize=True`).

### 3.3 Checkpoint State Dictionaries & Metadata
All 5 checkpoints residing in `backend/weights/` were inspected:
- `backend/weights/best_restormer.pt` (Epoch 3, Val Loss: 0.13149, Val PSNR: 18.32 dB, Val MS-SSIM: 0.8175)
- `backend/weights/last_checkpoint.pt` (Epoch 3, identical weights)
- `backend/weights/checkpoints/Restormer_RDB_epoch_001.pt` (Epoch 1, Val Loss: 0.17228, Val PSNR: 16.68 dB)
- `backend/weights/checkpoints/Restormer_RDB_epoch_002.pt` (Epoch 2, Val Loss: 0.15060, Val PSNR: 17.52 dB)
- `backend/weights/checkpoints/Restormer_RDB_epoch_003.pt` (Epoch 3, Val Loss: 0.13149, Val PSNR: 18.32 dB)
- **Key Sanitization**: All checkpoints deserialize clean 70 parameter tensors for `Restormer_RDB` with **zero missing keys** and **zero unexpected keys**. State dict keys require standard prefix sanitization (`_orig_mod.`, `module.`) and exclusion of profiling metadata (`total_ops`, `total_params`).
- **No EMA Weights**: None of the checkpoints contain exponential moving average weights (`model_ema` or `ema`).

### 3.4 Dataset Pairing & Split Protocol (`backend/train_or_eval.py`)
- **Dataset Inventory**:
  - `archive/data/original`: 858 images
  - `archive/data/corrected/protanopia`: 858 images
  - `archive/data/corrected/deuteranopia`: 858 images
  - `archive/data/corrected/tritanopia`: 858 images
  - Total verified condition pairs: $858 \times 3 = 2,574$ pairs.
- **Split Configuration**:
  - Seed: 2026.
  - Grouped strictly by image identity (image stem) before condition pairing.
  - Training (70%): 601 identities (1,803 pairs).
  - Validation (15%): 129 identities (387 pairs).
  - Testing (15%): 128 identities (384 pairs).
  - **Zero Data Leakage**: Automated test assertion confirms that the intersection of identities between train, val, and test splits is $\emptyset$.

### 3.5 Backend API & Personalization (`backend/app/`)
- `POST /api/v1/recolor`: Accepts Base64 image, `condition_id` $\in \{0, 1, 2\}$, `severity` $\in [0.0, 1.0]$, and `return_format` $\in \{\text{"webp"}, \text{"png"}, \text{"jpeg"}\}$.
- `GET /health`: Returns service health, loaded architecture name, device, and active cache size.
- `GET /api/v1/plates`: Returns 5 non-clinical screening plate definitions.
- `POST /api/v1/diagnose`: Evaluates plate responses and returns diagnosed CVD type and suggested severity.
- `POST /api/v1/filter`: Evaluates `page_filter.py` and returns a 20-element affine color matrix.
- `cache.py`: In-memory thread-safe LRU cache (capacity 500) using a composite key:
  $$\text{SHA256}(\text{raw\_bytes}) + \text{condition\_id} + \text{round}(\text{severity}, 3) + \text{model\_version} + \text{return\_format}$$
- `personalization.py`: Applies alpha blending between model prediction and original image:
  $$\mathbf{I}_{\text{final}} = \alpha \mathbf{I}_{\text{model}} + (1 - \alpha) \mathbf{I}_{\text{original}}, \quad \alpha = \text{clip}(\text{severity}, 0.0, 1.0)$$
  Severity 0 fast-path returns the original image immediately without invoking PyTorch inference.

### 3.6 Chrome Extension (Manifest V3)
- `manifest.json`: Manifest V3 compliant. Permissions: `storage`, `activeTab`. Host permissions: `http://localhost:8000/*`, `http://127.0.0.1:8000/*`, `<all_urls>`.
- `background.js`: Service worker handles cross-origin image fetching (`fetchImageAsBase64`), converts blobs to Base64, and proxies inference requests to `POST http://localhost:8000/api/v1/recolor`. Includes in-flight request deduplication via a `Map<cacheKey, Promise>`.
- `content.js`: Monitors DOM images using `MutationObserver`, filters out small icons ($< 32 \times 32$), SVG data URLs, and images inside sensitive password/credit-card forms. Replaces `img.src` while preserving layout dimensions (`width`, `height`, `style`). Stores original sources in `data-cvd-orig-src` for clean non-destructive restoration when disabled.
- `popup.html` / `popup.js`: Two-tab interface:
  1. Personalization: Enable/disable toggle, condition select (Protanopia / Deuteranopia / Tritanopia), severity slider ($0\% - 100\%$), recoloring scope selector, backend health indicator.
  2. Ishihara Test: 5-plate or 10-plate interactive screening test using dynamic CIE L*a*b* dot mosaics, error classification, and automatic profile configuration.

---

## 4. Discrepancy Matrix: Audit Claims vs. Actual Implementation

The historical `audit.txt` is an invaluable log of the project's evolution, but comparing its claims against the current codebase revealed several critical architectural and functional discrepancies:

| Topic | Claim in `audit.txt` | Actual Current Implementation | Engineering Impact |
|---|---|---|---|
| **Extension Recoloring Mechanism** | Section 5 claims extension was upgraded from full-page SVG filter to per-image deep neural recoloring. | In `content.js` (lines 26 & 488), `scope: "full_page"` is the **hardcoded default**. In `full_page` mode, it applies the SVG filter to `<html>` and **disconnects the image observer**, completely bypassing Restormer per-image neural inference! | The user sees an affine matrix color shift rather than the actual Restormer model's neural correction unless they manually switch to "Photos Only" in the popup. |
| **Transport Format & Color Mismatch** | Audit claims valid recoloring for WebP, PNG, and JPEG. | `background.js` (line 86) hardcodes `return_format: "webp"`. In `postprocessing.py`, WebP encoding uses lossy `quality=90`, which introduces 4:2:0 chroma subsampling and DCT compression noise. | The pixels rendered in the browser diverge from the Python PyTorch reference tensor due to lossy WebP compression. Lossless PNG transport must be used for reference equivalence. |
| **Reference Inference Pipeline** | Implicitly assumed uniform inference across modules. | No standalone `backend/app/core/reference_inference.py` exists. `predictor.py`, `train_or_eval.py`, and `page_filter.py` all implement their own separate inference invocations. | Risk of subtle divergence in padding, rounding, or condition broadcasting between training, testing, and production serving. |
| **Golden Reference Directory** | Audit references benchmark results and reference outputs. | `backend/reference_outputs/` does not exist yet. No static golden images or reproducibility manifests are checked in. | No baseline regression images exist to verify that future updates or optimizations preserve the model's exact color output. |
| **Dataset Linkage** | Audit states `archive/data/` is referenced by symlink `./dataset`. | On Windows, `./dataset` is a 12-byte plain text file containing `archive/data`, not an NTFS junction or symbolic link. | Python scripts had to implement custom path resolution fallbacks to find `archive/data`. |
| **Responsive Image Replacement (`srcset`)** | Audit claims content script replaces image sources without layout shifts. | `content.js` only updates `img.src`. It does not handle or clear `srcset`. | On modern responsive websites with `srcset`, the browser continues rendering the original responsive source, ignoring the recolored `src`. |
| **Candidate Checkpoints in Repo** | Mentions `Modified_Conditional_Restormer_seed_123_best.pt` in Google Drive. | That file is not in the workspace or local drives; only `Restormer_RDB` epoch 1, 2, 3 checkpoints exist locally in `backend/weights/`. | Local evaluation must focus on the locally available weights (`Restormer_RDB` epochs 1-3). |
| **Automated Testing Scope** | Audit claims 29/29 tests pass. | 29/29 tests do pass, but they test API status codes, schema validation, and cache hits. None test numerical color equivalence between PyTorch tensors and browser-decoded images. | Lack of regression tests that catch color divergence between the neural network and the extension. |

---

## 5. Verification of Existing Test Suite

The existing test suite was executed in the clean Python 3.11 virtual environment:
```powershell
.\backend\venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py" -v
```
**Result**:
- `test_backend.py`: 5 tests PASSED.
- `test_master_suite.py`: 24 tests PASSED.
- **Total: 29 / 29 tests PASSED in 2.13s.**
- Preserving 100% pass rate on these tests is an invariant requirement throughout all future phases.

---

## 6. Phase A Conclusion & Readiness

The project is structurally intact, healthy, and fully reproducible. The root cause of the perceived "model-to-extension color mismatch" has been pinpointed:
1. The default extension mode (`scope = "full_page"`) applies an affine least-squares SVG approximation rather than calling the neural model for images.
2. When image recoloring is called, the extension requests lossy WebP (`quality=90`), introducing chroma subsampling and lossy compression artifacts.
3. Lack of an authoritative, unified reference inference pipeline and golden output suite has allowed these subtle discrepancies to persist unnoticed.

Phase A inspection is complete. We are ready to proceed with Phase B (Model Checkpoint Discovery & Evaluation).

