from dataclasses import dataclass
import os
from pathlib import Path


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    aws_region: str
    s3_bucket: str
    s3_key: str
    download_dir: Path
    model_weights: str
    confidence_threshold: float
    frame_stride: int

    @property
    def local_video_path(self) -> Path:
        return self.download_dir / Path(self.s3_key).name


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    required_values = {
        "AWS_REGION": os.getenv("AWS_REGION"),
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "S3_KEY": os.getenv("S3_KEY"),
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    return AppConfig(
        aws_region=required_values["AWS_REGION"],
        s3_bucket=required_values["S3_BUCKET"],
        s3_key=required_values["S3_KEY"],
        download_dir=Path(os.getenv("DOWNLOAD_DIR", "downloads")),
        model_weights=os.getenv("YOLO_MODEL_WEIGHTS", "yolov8n.pt"),
        confidence_threshold=_read_float("CONFIDENCE_THRESHOLD", 0.25),
        frame_stride=_read_positive_int("FRAME_STRIDE", 1),
    )


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc

    if value < 1:
        raise ConfigError(f"{name} must be greater than or equal to 1")

    return value
