"""Configuration for the RAG lab, with a tiny zero-dependency .env loader."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _root() -> Path:
    # src/rag_lab/config.py -> repository root
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file if present (does not overwrite)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    docs_dir: Path = field(default_factory=lambda: _root() / "data" / "sample_docs")
    eval_path: Path = field(default_factory=lambda: _root() / "eval" / "questions.json")
    index_dir: Path = field(default_factory=lambda: _root() / "artifacts")
    logs_dir: Path = field(default_factory=lambda: _root() / "logs")
    chunk_size: int = 80        # words per chunk
    chunk_overlap: int = 20     # words of overlap between chunks
    top_k: int = 3
    log_query_text: bool = True  # False -> store only a SHA-256 hash of the query

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv(_root() / ".env")
        cfg = cls()
        if os.getenv("RAG_CHUNK_SIZE"):
            cfg.chunk_size = int(os.environ["RAG_CHUNK_SIZE"])
        if os.getenv("RAG_CHUNK_OVERLAP"):
            cfg.chunk_overlap = int(os.environ["RAG_CHUNK_OVERLAP"])
        if os.getenv("RAG_TOP_K"):
            cfg.top_k = int(os.environ["RAG_TOP_K"])
        if os.getenv("RAG_LOG_QUERY_TEXT"):
            cfg.log_query_text = _as_bool(os.environ["RAG_LOG_QUERY_TEXT"])
        return cfg
