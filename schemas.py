from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}


class TabularInput(BaseModel):
    features: list[float] = Field(..., min_length=1)
    feature_names: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def names_match_features(self) -> "TabularInput":
        if self.feature_names is not None and len(self.feature_names) != len(self.features):
            raise ValueError("`feature_names` length must match `features` length")
        return self


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=8_192)
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")


class ImageInput(BaseModel):
    data: str = Field(...)
    format: str = Field(default="jpeg", pattern=r"^(jpeg|png|webp)$")
    width: int | None = Field(default=None, gt=0, le=8_192)
    height: int | None = Field(default=None, gt=0, le=8_192)


class PredictRequest(_Base):
    input: TabularInput | TextInput | ImageInput = Field(..., discriminator=None)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def options_keys_are_strings(cls, v: dict) -> dict:
        if not all(isinstance(k, str) for k in v):
            raise ValueError("All option keys must be strings")
        return v


class PredictResponse(_Base):
    request_id: str
    predictions: list[Any]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(_Base):
    inputs: list[TabularInput | TextInput | ImageInput] = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class BatchPredictResponse(_Base):
    request_id: str
    results: list[PredictResponse]
    errors: list[str] = Field(default_factory=list)


class HealthResponse(_Base):
    status: str
    model_loaded: bool
    model_name: str
    go_backend_url: str


class ErrorResponse(_Base):
    error: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
