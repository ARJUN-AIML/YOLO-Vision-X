from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CAMERA_INDEX: int | str = 0 # Default to camera 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 640
    TARGET_FPS: int = 60
    # Upgrade to YOLOv8 Nano Segmentation for fast speed and perfect object contouring
    YOLO_MODEL: str = "yolov8n-seg.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    DEVICE: str = "cpu"

settings = Settings()