from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, BaseModel


class MainPath(BaseModel):
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR.joinpath("data")
    PROMPT_DIR: Path = BASE_DIR.joinpath("data", "prompts")


class OpenAiConfig(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file: str = ".env"


paths = MainPath()
openai_settings = OpenAiConfig()
