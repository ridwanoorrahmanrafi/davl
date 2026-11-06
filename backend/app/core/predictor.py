"""
Inference Pipeline and Model Management for Personalized CVD Recoloring.
"""

import hashlib
import time
from pathlib import Path
from threading import RLock
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

from .model import create_model, count_trainable_parameters
from .preprocessing import decode_base64_image
from .postprocessing import encode_base64_image, SUPPORTED_FORMATS
from .reference_inference import (
    preprocess_image_to_tensor,
    unpad_and_clamp_output,
    apply_severity_blend,
    resolve_condition,
)
from .cache import recolor_cache


class CVDRecolorPredictor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "Restormer_RDB",
        model_version: str = "1.0.0",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.model_version = f"{model_name}-v{model_version}"
        self.lock = RLock()
        self.fingerprint = "unloaded"
        self.weights_loaded = False

        # Instantiate production model architecture
        self.model = create_model(model_name=self.model_name)
        self.param_count = count_trainable_parameters(self.model)

        if weights_path and Path(weights_path).is_file():
            self.load_weights(weights_path)
        else:
            print(f"[CVDRecolorPredictor] Warning: No trained weights found at {weights_path}. Model initialized with random weights for integration testing.")

        self.model.to(self.device).eval()

    def load_weights(self, weights_path: str) -> bool:
        """Loads and verifies trained PyTorch checkpoint weights with architecture auto-detection."""
        try:
            path_obj = Path(weights_path)
            if not path_obj.is_file():
                self.weights_loaded = False
                return False

            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
            saved_model_name = checkpoint.get("architecture") or checkpoint.get("model_name")
            alias_map = {
                "CR_CBAM": "Restormer_CBAM",
                "Restormer_CBAM": "CR_CBAM",
                "Conditional_Restormer_Base": "Base_Restormer_Style",
                "Base_Restormer_Style": "Conditional_Restormer_Base",
                "CR_CBAM_RDB": "Modified_Conditional_Restormer",
                "Modified_Conditional_Restormer": "CR_CBAM_RDB",
            }

            if saved_model_name and saved_model_name != self.model_name and alias_map.get(saved_model_name) != self.model_name:
                self.weights_loaded = False
                print(f"[CVDRecolorPredictor] Error: Expected {self.model_name}, received {saved_model_name}")
                return False

            # Clean prefixes (_orig_mod., module., total_ops, total_params) and cast FP16 to FP32
            cleaned_state_dict = {}
            for k, v in raw_state_dict.items():
                if "total_ops" in k or "total_params" in k:
                    continue
                clean_k = k
                for prefix in ("_orig_mod.", "module."):
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    cleaned_state_dict[clean_k] = v.float()
                else:
                    cleaned_state_dict[clean_k] = v

            # Verify layer compatibility with self.model_name if no metadata present
            if not saved_model_name:
                has_cbam = any("cbam" in k for k in cleaned_state_dict.keys())
                has_rdb = any("rdb" in k for k in cleaned_state_dict.keys())
                is_cbam_arch = self.model_name in ("CR_CBAM", "Restormer_CBAM")
                is_rdb_arch = self.model_name == "Restormer_RDB"

                if (has_cbam and not has_rdb and not is_cbam_arch) or (has_rdb and not has_cbam and not is_rdb_arch):
                    # Auto-reconfigure architecture if unpinned
                    new_arch = "CR_CBAM" if (has_cbam and not has_rdb) else "Restormer_RDB"
                    print(f"[CVDRecolorPredictor] Auto-adapting model architecture to {new_arch}")
                    self.model_name = new_arch
                    self.model = create_model(model_name=self.model_name)
                    self.param_count = count_trainable_parameters(self.model)

            self.model.load_state_dict(cleaned_state_dict, strict=True)
            self.model.to(self.device).eval()
            self.weights_loaded = True
            self.fingerprint = hashlib.sha256(path_obj.read_bytes()).hexdigest()
            recolor_cache.clear()
            print(f"[CVDRecolorPredictor] Successfully loaded trained weights ({self.model_name}, {self.param_count:,} params) from: {weights_path}")
            return True
        except Exception as exc:
            self.weights_loaded = False
            self.fingerprint = "failed"
            print(f"[CVDRecolorPredictor] Error loading weights from {weights_path}: {exc}")
            return False

    @torch.inference_mode()
    def recolor(
        self,
        image_base64: str,
        condition_id_or_name: Any,
        severity: float = 1.0,
        return_format: str = "png",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end personalized recoloring:
        Decode -> Cache check -> Preprocessing -> Model inference -> Blending -> Postprocessing.
        """
        if not self.weights_loaded:
            raise RuntimeError("Trained Restormer_RDB checkpoint is required")

        start_total = time.perf_counter()

        condition_id = resolve_condition(condition_id_or_name)
        severity = float(np.clip(severity, 0.0, 1.0))
        return_format = return_format.strip().lower()
        if return_format == "jpg":
            return_format = "jpeg"

        # 1. Decode & validate input image
        pil_image, raw_bytes = decode_base64_image(image_base64)
        orig_np = np.array(pil_image, dtype=np.uint8)

        # 2. Check composite cache
        cache_key = recolor_cache.generate_key(
            raw_bytes,
            condition_id=condition_id,
            severity=severity,
            model_version=self.model_version,
            return_format=return_format,
        )
        cached_data_url = recolor_cache.get(cache_key)
        if cached_data_url is not None:
            total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0
            return {
                "status": "success",
                "recolored_base64": cached_data_url,
                "inference_time_ms": 0.0,
                "total_time_ms": round(total_elapsed_ms, 2),
                "cached": True,
                "condition_id": condition_id,
                "severity": severity,
                "model_version": self.model_version,
                "device": str(self.device),
            }

        # 3. Severity == 0 Fast-Path: return original image directly
        if severity == 0.0:
            encoded_image = encode_base64_image(orig_np, format_type=return_format)
            recolor_cache.put(cache_key, encoded_image)
            total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0
            return {
                "status": "success",
                "recolored_base64": encoded_image,
                "inference_time_ms": 0.0,
                "total_time_ms": round(total_elapsed_ms, 2),
                "cached": False,
                "condition_id": condition_id,
                "severity": severity,
                "model_version": self.model_version,
                "device": str(self.device),
            }

        # 4. Prepare normalized input tensor with spatial padding
        tensor, original_size, _ = preprocess_image_to_tensor(orig_np, device=self.device)

        # 5. Execute model inference (timed separately)
        cond_tensor = torch.tensor([condition_id], dtype=torch.long, device=self.device)

        t_inf_start = time.perf_counter()
        with self.lock:
            pred_tensor = self.model(tensor, cond_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        t_inf_end = time.perf_counter()
        inference_time_ms = (t_inf_end - t_inf_start) * 1000.0

        # 6. Unpad and clamp model output
        _, recolored_np = unpad_and_clamp_output(pred_tensor, original_size)

        # 7. Apply Personalization Blending
        # Output = severity * ModelCorrection + (1 - severity) * Original
        final_np = apply_severity_blend(orig_np, recolored_np, severity)

        # 8. Encode to requested format (WebP / PNG / JPEG)
        encoded_image = encode_base64_image(final_np, format_type=return_format)

        # 9. Store in cache
        recolor_cache.put(cache_key, encoded_image)

        total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0

        return {
            "status": "success",
            "recolored_base64": encoded_image,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_elapsed_ms, 2),
            "cached": False,
            "condition_id": condition_id,
            "severity": severity,
            "model_version": self.model_version,
            "device": str(self.device),
        }
