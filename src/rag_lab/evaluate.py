"""Retrieval evaluation harness.

Runs a labelled question set through the retriever and reports hit@1,
recall@k, and MRR. Every query is written to the audit log.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .audit import AuditLog
from .config import Config
from .ingest import build_chunks, load_documents
from .retriever import TfidfRetriever


def load_eval(path: Path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(cfg: Config | None = None) -> Tuple[Dict[str, float], List[dict]]:
    cfg = cfg or Config.from_env()

    documents = load_documents(cfg.docs_dir)
    chunks = build_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)
    retriever = TfidfRetriever().fit(chunks)
    audit = AuditLog(cfg.logs_dir / "audit.jsonl", cfg.log_query_text)

    questions = load_eval(cfg.eval_path)
    total = len(questions)
    recall_key = f"recall@{cfg.top_k}"

    hits_at_1 = 0
    recall_hits = 0
    reciprocal_ranks = 0.0
    per_question: List[dict] = []

    for item in questions:
        query = item["question"]
        relevant = set(item["relevant_docs"])
        results = retriever.query(query, top_k=cfg.top_k)
        audit.log_query(query, results)

        retrieved_docs = [r.doc_id for r in results]
        hit_at_1 = bool(retrieved_docs) and retrieved_docs[0] in relevant
        recall = any(d in relevant for d in retrieved_docs)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break

        hits_at_1 += int(hit_at_1)
        recall_hits += int(recall)
        reciprocal_ranks += rr
        per_question.append({
            "question": query,
            "expected": sorted(relevant),
            "retrieved": retrieved_docs,
            "hit@1": hit_at_1,
            recall_key: recall,
            "rr": round(rr, 3),
        })

    metrics = {
        "questions": total,
        "hit@1": round(hits_at_1 / total, 3) if total else 0.0,
        recall_key: round(recall_hits / total, 3) if total else 0.0,
        "mrr": round(reciprocal_ranks / total, 3) if total else 0.0,
    }
    return metrics, per_question


def main() -> None:
    cfg = Config.from_env()
    recall_key = f"recall@{cfg.top_k}"
    metrics, per_question = evaluate(cfg)

    print("Per-question results:")
    for row in per_question:
        status = "OK  " if row[recall_key] else "MISS"
        print(f"  [{status}] {row['question']}")
        print(f"          expected={row['expected']} retrieved={row['retrieved']}")

    print("\nAggregate metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"\nAudit log written to: {cfg.logs_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
