"""Small atomic cache helpers used by resumable data preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Replace a text artifact only after the complete temporary write succeeds."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    return target
