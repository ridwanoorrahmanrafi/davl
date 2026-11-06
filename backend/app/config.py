"""
Backend Configuration for Personalized CVD Recoloring API.
"""

from pathlib import Path
import os


class Settings:
    PROJECT_NAME: str = "Personalized CVD Recoloring API"
    VERSION: str = "1.0.0"
    HOST: str = os.environ.get("CVD_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("CVD_PORT", "8000"))

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"

    # Default checkpoint detection:
    # 1. Explicit environment variable override (CVD_WEIGHTS)
    # 2. Winning ablation model from validation-first notebook: CR_CBAM (seed 3407)
    # 3. Reference benchmark model: Restormer_RDB (epoch 3)
    _ENV_WEIGHTS = os.environ.get("CVD_WEIGHTS")
    _ENV_MODEL = os.environ.get("CVD_MODEL_NAME")

    _CBAM_CANDIDATES = [
        WEIGHTS_DIR / "CR_CBAM_seed_3407_FP16_state_dict.pt",
        WEIGHTS_DIR / "best_cbam.pt",
        WEIGHTS_DIR / "CR_CBAM_seed_3407.pt",
    ]

    _DETECTED_CBAM = next((p for p in _CBAM_CANDIDATES if p.is_file()), None)

    if _ENV_WEIGHTS:
        DEFAULT_WEIGHTS_PATH: Path = Path(_ENV_WEIGHTS)
        PRODUCTION_MODEL_NAME: str = _ENV_MODEL or ("CR_CBAM" if "cbam" in str(_ENV_WEIGHTS).lower() else "Restormer_RDB")
    elif _DETECTED_CBAM:
        DEFAULT_WEIGHTS_PATH: Path = _DETECTED_CBAM
        PRODUCTION_MODEL_NAME: str = _ENV_MODEL or "CR_CBAM"
    else:
        DEFAULT_WEIGHTS_PATH: Path = WEIGHTS_DIR / "best_restormer.pt"
        PRODUCTION_MODEL_NAME: str = _ENV_MODEL or "Restormer_RDB"

    # Safety: Require trained checkpoint in strict production mode
    REQUIRE_TRAINED_WEIGHTS: bool = os.environ.get("REQUIRE_TRAINED_WEIGHTS", "0").lower() in ("1", "true", "yes")

    # Hardware & Performance
    DEVICE: str = os.environ.get("CVD_DEVICE", "cuda")  # Auto-falls back to cpu if cuda unavailable
    MAX_CACHE_SIZE: int = int(os.environ.get("CVD_CACHE_SIZE", "500"))


settings = Settings()
