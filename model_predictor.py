"""Utilities for loading trained aroma prediction models.

This helper encapsulates the directory layout produced by ``ml_build_plots.py``.
It exposes a small API that lets consumers list available targets and load the
serialized estimators (together with the scaler and metadata) for a specific
sensory attribute.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class FlavorPredictor:
    """Loader for persisted models exported by ``ml_build_plots.py``.

    Parameters
    ----------
    models_root:
        Path to the ``saved_models`` directory created by the training script.
    """

    def __init__(self, models_root: Path | str) -> None:
        self.models_root = Path(models_root)
        if not self.models_root.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_root}")
        if not self.models_root.is_dir():
            raise NotADirectoryError(f"Models path is not a directory: {self.models_root}")

        # Cache already-loaded targets so repeated calls do not repeat disk I/O.
        self._models_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available_targets(self) -> Iterable[str]:
        """Return a sorted iterable of target (sensory attribute) names."""

        return sorted(p.name for p in self.models_root.iterdir() if p.is_dir())

    def load_models_for_target(self, target: str) -> Dict[str, Dict[str, Any]]:
        """Load all persisted models for ``target``.

        The returned dictionary is keyed by the human-readable model name
        (e.g. ``"Lasso Regression"``) and each entry contains:

        ``model``
            The fitted estimator ready for inference.
        ``scaler``
            The ``StandardScaler`` instance used during training.
        ``feature_columns``
            Ordered list of feature names expected by the model/scaler.
        ``params``
            Hyper-parameters selected during tuning.
        ``performance``
            Dictionary with the stored R^2, RMSE and MAE metrics.
        ``metadata``
            Any additional information persisted in the pickle payload.
        """

        if target in self._models_cache:
            return self._models_cache[target]

        target_dir = self._target_dir(target)

        summary, preloaded_payloads = self._read_summary(target_dir)
        if not summary:
            summary, preloaded_payloads = self._build_summary_fallback(target_dir)

        models: Dict[str, Dict[str, Any]] = {}
        for model_name, meta in summary.items():
            payload = preloaded_payloads.get(model_name)
            if payload is None:
                model_path = self._resolve_model_path(target_dir, model_name, meta)
                payload = self._read_pickle(model_path)

            models[model_name] = self._assemble_model_dict(payload)

        self._models_cache[target] = models
        return models

    def load_single_model(self, target: str, model_name: str) -> Dict[str, Any]:
        """Convenience helper that returns one model descriptor."""

        models = self.load_models_for_target(target)
        if model_name not in models:
            available = ", ".join(sorted(models)) or "<none>"
            raise KeyError(
                f"Model '{model_name}' not found for target '{target}'. "
                f"Available: {available}"
            )
        return models[model_name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _target_dir(self, target: str) -> Path:
        candidate = self.models_root / target
        if not candidate.exists():
            available = ", ".join(self.available_targets()) or "<none>"
            raise FileNotFoundError(
                f"Target '{target}' not found in {self.models_root}. "
                f"Available targets: {available}"
            )
        return candidate

    def _read_summary(self, target_dir: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        summary_path = target_dir / "models_summary.json"
        if not summary_path.exists():
            return {}, {}

        with summary_path.open(encoding="utf-8") as fh:
            raw_summary = json.load(fh)

        # Normalise keys to strings (JSON decoder already returns str, but this keeps typing happy).
        summary: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in raw_summary.items()}
        return summary, {}

    def _build_summary_fallback(self, target_dir: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        summary: Dict[str, Dict[str, Any]] = {}
        preloaded: Dict[str, Dict[str, Any]] = {}

        for pkl_path in target_dir.glob("*.pkl"):
            payload = self._read_pickle(pkl_path)
            model_name = payload.get("model_name") or pkl_path.stem.replace("_", " ")
            summary[model_name] = {"file_path": str(pkl_path)}
            preloaded[model_name] = payload

        return summary, preloaded

    def _resolve_model_path(self, target_dir: Path, model_name: str, meta: Dict[str, Any]) -> Path:
        candidate: Optional[str] = None
        if meta:
            candidate = meta.get("file_path") or meta.get("path")

        if candidate:
            model_path = Path(candidate)
            if not model_path.exists():
                # Try interpreting the path as relative to the target directory.
                model_path = target_dir / Path(candidate).name
        else:
            sanitized = model_name.replace(" ", "_") + "_model.pkl"
            model_path = target_dir / sanitized

        if not model_path.exists():
            raise FileNotFoundError(
                f"Serialized model for '{model_name}' not found. Looked at: {model_path}"
            )
        return model_path

    def _read_pickle(self, path: Path) -> Dict[str, Any]:
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _assemble_model_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        performance = {
            "R2": payload.get("R2"),
            "RMSE": payload.get("RMSE"),
            "MAE": payload.get("MAE"),
        }
        core_keys = {"model", "scaler", "feature_columns", "params", "R2", "RMSE", "MAE"}
        metadata = {k: v for k, v in payload.items() if k not in core_keys}

        return {
            "model": payload.get("model"),
            "scaler": payload.get("scaler"),
            "feature_columns": payload.get("feature_columns"),
            "params": payload.get("params"),
            "performance": performance,
            "metadata": metadata,
        }


__all__ = ["FlavorPredictor"]
