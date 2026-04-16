from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_view: bool = True

    img_size: int = 224
    bbox_offset: int = 20

    gesture_delay_s: float = 1.0
    ml_delay_s: float = 1.5
    confidence_threshold: float = 0.95

    assets_dir: Path = Path(__file__).resolve().parent / "assets"
