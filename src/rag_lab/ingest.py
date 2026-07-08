"""Ingestion pipeline: load local documents and split them into chunks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """Return a list of (doc_id, text) for every Markdown file in docs_dir."""
    documents: List[Tuple[str, str]] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        documents.append((path.stem, path.read_text(encoding="utf-8")))
    return documents


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(documents, chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc_id, text in documents:
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, text=piece))
    return chunks


def main() -> None:
    from .config import Config

    cfg = Config.from_env()
    documents = load_documents(cfg.docs_dir)
    chunks = build_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.index_dir / "chunks.json"
    out_path.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8"
    )

    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
