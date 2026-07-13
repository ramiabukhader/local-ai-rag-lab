"""Append-only JSONL audit log for retrieval queries.

Every query is recorded so a retrieval pipeline over business documents is
traceable. The default stores only a SHA-256 hash; raw query text requires an
explicit ``log_query_text=True`` opt-in.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


class AuditLog:
    def __init__(self, log_path: Path, log_query_text: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_query_text = log_query_text

    def log_query(self, query: str, results: List) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "result_count": len(results),
            "retrieved": [
                {"chunk_id": r.chunk_id, "doc_id": r.doc_id, "score": round(r.score, 4)}
                for r in results
            ],
        }
        if self.log_query_text:
            entry["query"] = query
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry
