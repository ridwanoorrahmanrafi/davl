"""
Authoritative Reference Inference Pipeline for DVAL Restormer_RDB.
Single canonical implementation for all backend, testing, and reference generation components.

Guarantees:
- Expected input: RGB image, float32, range [0.0, 1.0], tensor shape (B, 3, H, W).
- Condition IDs: 0 = Protanopia, 1 = Deuteranopia, 2 = Tritanopia.
- Spatially broadcasted one-hot condition channels concatenated along dim 1 -> (B, 6, H, W).
- Bounded residual output formulation:
    residual = 0.5 * torch.tanh(output_projection(decoded))
    prediction = torch.clamp(original_image + residual, 0.0, 1.0)
- Spatial padding to multiples of 8 with mode="replicate", with exact unpadding back to original (H, W).
- Uses torch.inference_mode(), model.eval(), thread-safe execution.
"""

import hashlib
from pathlib import Path
from threading import RLock
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .model import create_model, append_condition_channels, count_trainable_parameters

EXPECTED_CHECKPOINT_SHA256 = "7f5ad90b372028d17b09fcaaa97b039d824e13d434a09814448af4bca6383eed"
EXPECTED_PARAM_COUNT = 461694

CONDITION_MAP = {
    0: "Protanopia",
    1: "Deuteranopia",
    2: "Tritanopia",
}

NAME_TO_CONDITION_ID = {
    "protanopia": 0,
    "protan": 0,
    "0": 0,
    "deuteranopia": 1,
    "deutan": 1,
    "1": 1,
    "tritanopia": 2,
    "tritan": 2,
    "2": 2,
}


def resolve_condition(condition_id_or_name: Any) -> int:
    """Resolves condition integer or string to validated ID {0, 1, 2}."""
    if isinstance(condition_id_or_name, int):
        if condition_id_or_name in CONDITION_MAP:
            return condition_id_or_name
        raise ValueError(f"Invalid condition_id: {condition_id_or_name}. Expected 0 (Protanopia), 1 (Deuteranopia), or 2 (Tritanopia).")

    if isinstance(condition_id_or_name, str):
        clean = condition_id_or_name.strip().lower()
        if clean in NAME_TO_CONDITION_ID:
            return NAME_TO_CONDITION_ID[clean]

    raise ValueError(f"Unknown condition: '{condition_id_or_name}'. Expected 0, 1, 2 or Protanopia, Deuteranopia, Tritanopia.")


def load_authoritative_reference_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    verify_hash: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads and strictly verifies the production reference Restormer_RDB model.
    Never silently falls back to random weights.
    """
    path_obj = Path(checkpoint_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Authoritative production checkpoint not found at: {path_obj}")

    file_bytes = path_obj.read_bytes()
    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

    if verify_hash and computed_sha256 != EXPECTED_CHECKPOINT_SHA256:
        print(f"[ReferenceInference] Warning: Checkpoint SHA-256 ({computed_sha256}) differs from expected reference ({EXPECTED_CHECKPOINT_SHA256}).")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path_obj, map_location=device, weights_only=True)
    raw_sd = ckpt.get("model_state_dict", ckpt)

    # Clean state dict keys and cast FP16 to FP32 if needed
    clean_sd = {}
    for k, v in raw_sd.items():
        if "total_ops" in k or "total_params" in k:
            continue
        clean_k = k
        for pfx in ("_orig_mod.", "module."):
            if clean_k.startswith(pfx):
                clean_k = clean_k[len(pfx):]
        if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            clean_sd[clean_k] = v.float()
        else:
            clean_sd[clean_k] = v

    has_cbam = any("cbam" in k for k in clean_sd.keys())
    has_rdb = any("rdb" in k for k in clean_sd.keys())

    arch = ckpt.get("architecture") or ckpt.get("model_name")
    if not arch:
        if has_cbam and not has_rdb:
            arch = "CR_CBAM"
        elif has_rdb and not has_cbam:
            arch = "Restormer_RDB"
        elif has_cbam and has_rdb:
            arch = "CR_CBAM_RDB"
        else:
            arch = "Conditional_Restormer_Base"

    model = create_model(arch).to(device)
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    param_count = count_trainable_parameters(model)

    metadata = {
        "architecture": arch,
        "checkpoint_path": str(path_obj),
        "checkpoint_sha256": computed_sha256,
        "parameter_count": param_count,
        "device": str(device),
        "epoch": ckpt.get("epoch", 3),
        "val_loss": ckpt.get("val_loss", None),
    }

    return model, metadata


def preprocess_image_to_tensor(
    image: Union[Image.Image, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Standardizes input image to normalized float32 tensor [0.0, 1.0],
    and pads spatial dimensions (H, W) to multiples of 8 using replicate padding.

    Returns:
        padded_tensor: (1, 3, H_pad, W_pad) in range [0.0, 1.0]
        original_size: (H, W)
        pad_amounts: (pad_h, pad_w)
    """
    if isinstance(image, Image.Image):
        # Correct EXIF orientation
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image, dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            # Drop alpha channel
            image = image[:, :, :3]
        if image.dtype == np.uint8:
            img_np = image.astype(np.float32) / 255.0
        else:
            img_np = np.clip(image.astype(np.float32), 0.0, 1.0)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    # (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = tensor.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    return tensor, (h, w), (pad_h, pad_w)


def unpad_and_clamp_output(
    pred_tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Crops output tensor back to exact original (H, W) and clamps to [0.0, 1.0].

    Returns:
        float_tensor: (1, 3, H, W) float32 tensor clamped to [0.0, 1.0]
        uint8_array: (H, W, 3) uint8 numpy array in range [0, 255]
    """
    orig_h, orig_w = original_size
    cropped = pred_tensor[:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
    uint8_np = (cropped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return cropped, uint8_np


def apply_severity_blend(
    original_uint8: np.ndarray,
    recolored_uint8: np.ndarray,
    severity: float = 1.0,
) -> np.ndarray:
    """
    Applies personalization blending:
    Final = severity * ModelCorrection + (1 - severity) * Original
    """
    alpha = float(np.clip(severity, 0.0, 1.0))
    if alpha == 0.0:
        return original_uint8.copy()
    if alpha == 1.0:
        return recolored_uint8.copy()
    blended = alpha * recolored_uint8.astype(np.float32) + (1.0 - alpha) * original_uint8.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).round().astype(np.uint8)


class AuthoritativeReferenceInference:
    """
    Singleton-friendly reference inference pipeline ensuring 100% reproducible execution.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "backend/weights/best_restormer.pt",
        device: Optional[str] = None,
    ):
        self.lock = RLock()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model, self.metadata = load_authoritative_reference_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            verify_hash=False,
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        condition: Any,
        severity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Executes reference inference.
        Returns:
            - "recolored_uint8": (H, W, 3) uint8 NumPy array of model correction (severity=1.0)
            - "blended_uint8": (H, W, 3) uint8 NumPy array after personalization blending
            - "original_uint8": (H, W, 3) uint8 NumPy array of preprocessed original image
            - "condition_id": integer {0, 1, 2}
            - "condition_name": string {"Protanopia", "Deuteranopia", "Tritanopia"}
            - "severity": float in [0.0, 1.0]
            - "original_size": (H, W)
        """
        condition_id = resolve_condition(condition)
        severity = float(np.clip(severity, 0.0, 1.0))

        # Standardize input
        if isinstance(image, Image.Image):
            orig_pil = ImageOps.exif_transpose(image).convert("RGB")
            orig_uint8 = np.array(orig_pil, dtype=np.uint8)
        else:
            orig_uint8 = image if image.dtype == np.uint8 else (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # Severity 0 fast path
        if severity == 0.0:
            return {
                "recolored_uint8": orig_uint8.copy(),
                "blended_uint8": orig_uint8.copy(),
                "original_uint8": orig_uint8,
                "condition_id": condition_id,
                "condition_name": CONDITION_MAP[condition_id],
                "severity": 0.0,
                "original_size": (orig_uint8.shape[0], orig_uint8.shape[1]),
                "checkpoint_sha256": self.metadata["checkpoint_sha256"],
            }

        # Preprocess to padded tensor
        padded_tensor, original_size, _ = preprocess_image_to_tensor(orig_uint8, device=self.device)
        cond_tensor = torch.tensor([condition_id], dtype=torch.long, device=self.device)

        # Forward pass
        with self.lock:
            pred_tensor = self.model(padded_tensor, cond_tensor)

        # Unpad and convert
        _, recolored_uint8 = unpad_and_clamp_output(pred_tensor, original_size)

        # Personalization blending
        blended_uint8 = apply_severity_blend(orig_uint8, recolored_uint8, severity)

        return {
            "recolored_uint8": recolored_uint8,
            "blended_uint8": blended_uint8,
            "original_uint8": orig_uint8,
            "condition_id": condition_id,
            "condition_name": CONDITION_MAP[condition_id],
            "severity": severity,
            "original_size": original_size,
            "checkpoint_sha256": self.metadata["checkpoint_sha256"],
        }

