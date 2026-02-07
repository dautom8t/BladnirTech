"""
AME ML Model Store

Persists trained models to disk with versioning, metadata, and integrity verification.
Models are stored as joblib pickles with companion JSON metadata and SHA-256 checksums.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

DEFAULT_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "models")
)


def _compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    """Persists trained models to disk with versioning and integrity verification."""

    def __init__(self, base_dir: str = DEFAULT_MODEL_DIR):
        self.base_dir = os.path.abspath(base_dir)

    def _scope_dir(self, scope_key: str) -> str:
        safe_key = scope_key.replace(":", "/").replace("..", "")
        return os.path.join(self.base_dir, safe_key)

    def save(self, model: Any, scope_key: str, model_type: str, metadata: Dict[str, Any]) -> int:
        """Save model to disk with SHA-256 checksum. Returns new version number."""
        if not HAS_JOBLIB:
            logger.warning("joblib not available, model not persisted")
            return 0

        directory = self._scope_dir(scope_key)
        os.makedirs(directory, exist_ok=True)

        version = self._next_version(directory, model_type)

        model_path = os.path.join(directory, f"{model_type}_v{version}.pkl")
        joblib.dump(model, model_path)

        # Compute SHA-256 checksum of the saved model file
        checksum = _compute_file_hash(model_path)

        metadata.update({
            "version": version,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "model_path": model_path,
            "sha256": checksum,
        })

        meta_path = os.path.join(directory, f"{model_type}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved {model_type} v{version} to {model_path} (sha256:{checksum[:12]}...)")
        return version

    def load(self, scope_key: str, model_type: str) -> Tuple[Any, Dict[str, Any]]:
        """Load latest model with integrity verification. Returns (model, metadata) or (None, {})."""
        if not HAS_JOBLIB:
            return None, {}

        directory = self._scope_dir(scope_key)
        meta_path = os.path.join(directory, f"{model_type}_metadata.json")

        if not os.path.exists(meta_path):
            return None, {}

        try:
            with open(meta_path) as f:
                metadata = json.load(f)

            version = metadata.get("version", 1)
            model_path = os.path.join(directory, f"{model_type}_v{version}.pkl")

            if not os.path.exists(model_path):
                return None, metadata

            # Verify SHA-256 checksum before loading
            expected_hash = metadata.get("sha256")
            if expected_hash:
                actual_hash = _compute_file_hash(model_path)
                if actual_hash != expected_hash:
                    logger.error(
                        "Model integrity check FAILED for %s v%s: "
                        "expected sha256=%s, got sha256=%s. "
                        "Refusing to load potentially tampered model.",
                        model_type, version, expected_hash[:12], actual_hash[:12],
                    )
                    return None, metadata
            else:
                logger.warning(
                    "No SHA-256 checksum found for %s v%s — "
                    "loading without integrity verification (legacy model).",
                    model_type, version,
                )

            model = joblib.load(model_path)
            logger.info(f"Loaded {model_type} v{version} from {model_path}")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            return None, {}

    def get_metadata(self, scope_key: str, model_type: str) -> Dict[str, Any]:
        """Get metadata without loading model."""
        directory = self._scope_dir(scope_key)
        meta_path = os.path.join(directory, f"{model_type}_metadata.json")

        if not os.path.exists(meta_path):
            return {}

        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _next_version(self, directory: str, model_type: str) -> int:
        meta_path = os.path.join(directory, f"{model_type}_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f).get("version", 0) + 1
            except Exception:
                pass
        return 1
