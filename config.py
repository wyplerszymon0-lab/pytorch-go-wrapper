from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    go_backend_url: str = Field(default="http://localhost:9090")
    go_timeout: float = Field(default=10.0, gt=0)
    go_preprocess_enabled: bool = Field(default=False)
    go_postprocess_enabled: bool = Field(default=False)

    api_key_required: bool = Field(default=False)
    api_key: str = Field(default="")

    model_name: str = Field(default="example-net")
    model_weights_path: str | None = Field(default=None)
    model_input_dim: int = Field(default=128, gt=0)
    model_num_classes: int = Field(default=10, gt=0)

    use_gpu: bool = Field(default=False)
    use_torch_compile: bool = Field(default=False)

    max_batch_size: int = Field(default=64, gt=0)
    batch_concurrency: int = Field(default=8, gt=0)

    allowed_origins: list[str] = Field(default=["*"])

    @field_validator("go_backend_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")
