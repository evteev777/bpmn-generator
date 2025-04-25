# config.py

from pathlib import Path

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: Path
    PROMPT_DIR: Path
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    USE_FP16: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
