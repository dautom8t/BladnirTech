"""
Storage abstraction layer.

Demo / development  →  local filesystem (``./data/files``)
Production          →  S3-compatible object store (when STORAGE_BACKEND=s3)

Usage::

    from services.storage import storage

    storage.put("reports/daily.json", json_bytes)
    data = storage.get("reports/daily.json")
    storage.delete("reports/daily.json")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# =====================================
# Abstract interface
# =====================================

class StorageBackend(ABC):
    """Minimal object-storage contract."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Return file contents or *None* if missing."""
        ...

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Write *data* under *key* (overwrites)."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete *key*.  Return ``True`` if it existed."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """Return keys matching *prefix*."""
        ...

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Backend metadata / health."""
        ...


# =====================================
# Local filesystem implementation (demo)
# =====================================

class LocalStorage(StorageBackend):
    """Store objects as plain files under a root directory."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage root: {self._root.resolve()}")

    def _path(self, key: str) -> Path:
        # Prevent directory traversal
        safe = Path(key).resolve()
        resolved_root = self._root.resolve()
        if not str(safe).startswith(str(resolved_root)):
            # Rebuild safely under root
            safe = self._root / Path(key).name
        else:
            safe = self._root / key
        return safe

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if not p.exists():
            return None
        return p.read_bytes()

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_keys(self, prefix: str = "") -> List[str]:
        results: List[str] = []
        for p in self._root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self._root))
                if rel.startswith(prefix):
                    results.append(rel)
        return sorted(results)

    def info(self) -> Dict[str, Any]:
        count = sum(1 for _ in self._root.rglob("*") if _.is_file())
        return {"backend": "local", "root": str(self._root.resolve()), "file_count": count}


# =====================================
# S3 stub (production placeholder)
# =====================================

class S3Storage(StorageBackend):
    """
    Placeholder for an S3-backed store.

    Requires ``boto3`` (``pip install boto3``).
    In demo mode this class is never instantiated.
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "S3 storage backend requires the 'boto3' package. "
                "Install it with: pip install boto3"
            ) from exc
        self._bucket = bucket
        self._s3 = boto3.client("s3", region_name=region)
        logger.info(f"S3Storage bucket: {bucket} ({region})")

    def get(self, key: str) -> Optional[bytes]:
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            return obj["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def delete(self, key: str) -> bool:
        self._s3.delete_object(Bucket=self._bucket, Key=key)
        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def info(self) -> Dict[str, Any]:
        return {"backend": "s3", "bucket": self._bucket}


# =====================================
# Factory
# =====================================

def _build_storage() -> StorageBackend:
    from config import settings

    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("S3 storage requires S3_BUCKET environment variable")
        return S3Storage(bucket=settings.s3_bucket, region=settings.s3_region)

    return LocalStorage(root=settings.storage_local_dir)


storage: StorageBackend = _build_storage()
logger.info(f"Storage initialised: {storage.info()['backend']}")
