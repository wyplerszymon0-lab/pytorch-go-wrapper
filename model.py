from __future__ import annotations

import logging
import threading
from typing import Any

import torch
import torch.nn as nn

from config import Settings

logger = logging.getLogger("model")


class _ExampleNet(nn.Module):
    def __init__(self, input_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        self._device = torch.device(
            "cuda" if self._settings.use_gpu and torch.cuda.is_available() else "cpu"
        )
        logger.info("Loading model '%s' on %s …", self._settings.model_name, self._device)

        model = _ExampleNet(
            input_dim=self._settings.model_input_dim,
            num_classes=self._settings.model_num_classes,
        )

        if self._settings.model_weights_path:
            state = torch.load(
                self._settings.model_weights_path,
                map_location=self._device,
                weights_only=True,
            )
            model.load_state_dict(state)
            logger.info("Weights loaded from %s", self._settings.model_weights_path)
        else:
            logger.warning("No weights path configured — using random weights (dev mode)")

        model.to(self._device)
        model.eval()

        if self._settings.use_torch_compile:
            try:
                model = torch.compile(model)  # type: ignore[assignment]
                logger.info("Model compiled with torch.compile ✓")
            except Exception as exc:
                logger.warning("torch.compile failed, continuing without: %s", exc)

        self._model = model
        self._warm_up()
        logger.info("Model ready ✓")

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, input_data: Any, go_context: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        tensor = self._preprocess(input_data, go_context)

        with self._lock:
            with torch.inference_mode():
                logits = self._model(tensor)

        return self._postprocess(logits, go_context)

    def _preprocess(self, input_data: Any, go_context: dict[str, Any]) -> torch.Tensor:
        from schemas import TabularInput, TextInput, ImageInput

        if isinstance(input_data, TabularInput):
            raw = torch.tensor(input_data.features, dtype=torch.float32)
        elif isinstance(input_data, TextInput):
            logger.debug("Text input — using stub embedding")
            raw = torch.zeros(self._settings.model_input_dim, dtype=torch.float32)
        elif isinstance(input_data, ImageInput):
            logger.debug("Image input — using stub tensor")
            raw = torch.zeros(self._settings.model_input_dim, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        if "scale_factor" in go_context:
            raw = raw * float(go_context["scale_factor"])

        if raw.shape[0] != self._settings.model_input_dim:
            raw = torch.nn.functional.pad(
                raw[:self._settings.model_input_dim],
                (0, max(0, self._settings.model_input_dim - raw.shape[0])),
            )

        return raw.unsqueeze(0).to(self._device)

    def _postprocess(self, logits: torch.Tensor, go_context: dict[str, Any]) -> dict[str, Any]:
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_conf, top_idx = probs.max(dim=-1)

        predictions = probs.tolist()
        confidence = float(top_conf)
        top_class = int(top_idx)

        metadata: dict[str, Any] = {
            "top_class": top_class,
            "device": str(self._device),
        }

        if "class_labels" in go_context:
            labels: list[str] = go_context["class_labels"]
            if top_class < len(labels):
                metadata["top_label"] = labels[top_class]

        return {
            "predictions": predictions,
            "confidence": confidence,
            "metadata": metadata,
        }

    def _warm_up(self, n: int = 3) -> None:
        dummy = torch.zeros(1, self._settings.model_input_dim, device=self._device)
        with torch.inference_mode():
            for _ in range(n):
                self._model(dummy)  # type: ignore[misc]
        logger.info("Warm-up complete (%d passes)", n)
