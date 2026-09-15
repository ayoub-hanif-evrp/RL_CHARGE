"""Train-only feature normalization. Validation and test statistics never enter fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


EPS = 1e-6


@dataclass
class Normalizer:
    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    fitted: bool = False

    @classmethod
    def empty(cls) -> "Normalizer":
        return cls(mean={}, std={}, fitted=False)

    def fit(self, batches: Mapping[str, np.ndarray]) -> "Normalizer":
        self.mean = {}
        self.std = {}
        for key, values in batches.items():
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                self.mean[key] = np.zeros((1,), dtype=np.float64)
                self.std[key] = np.ones((1,), dtype=np.float64)
                continue
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            last = array.shape[-1]
            flat = array.reshape(-1, last)
            self.mean[key] = flat.mean(axis=0)
            std = flat.std(axis=0)
            std = np.where(std < EPS, 1.0, std)
            self.std[key] = std
        self.fitted = True
        return self

    def transform(self, key: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not self.fitted or key not in self.mean:
            return array.astype(np.float32)
        mean = self.mean[key]
        std = self.std[key]
        original = array.shape
        last = mean.shape[0]
        if array.ndim == 1:
            array = array.reshape(1, -1)
        flat = array.reshape(-1, last)
        out = (flat - mean) / std
        return out.reshape(original).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": {key: value.tolist() for key, value in self.mean.items()},
            "std": {key: value.tolist() for key, value in self.std.items()},
            "fitted": self.fitted,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "Normalizer":
        return cls(
            mean={key: np.asarray(value, dtype=np.float64) for key, value in payload["mean"].items()},
            std={key: np.asarray(value, dtype=np.float64) for key, value in payload["std"].items()},
            fitted=bool(payload.get("fitted", True)),
        )
