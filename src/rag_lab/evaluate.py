"""Retrieval evaluation harness.

Runs a labelled question set through the retriever and reports hit@1,
recall@k, and MRR. Every query is written to the audit log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .audit import AuditLog
from .config import Config
from .ingest import build_chunks, load_documents
from .retriever import TfidfRetriever


class DatasetValidationError(ValueError):
    """An actionable, content-safe evaluation dataset error."""


def load_eval(path: Path, available_doc_ids: Optional[Iterable[str]] = None) -> List[dict]:
    dataset_path = Path(path)
    try:
        raw = dataset_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetValidationError(
            f"cannot read evaluation dataset {dataset_path}: {error.strerror or error}"
        ) from error
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"invalid JSON in {dataset_path} at line {error.lineno}, column {error.colno}"
        ) from error

    if not isinstance(questions, list):
        raise DatasetValidationError(f"{dataset_path}: top-level value must be an array")
    if not questions:
        raise DatasetValidationError(f"{dataset_path}: dataset must contain at least one question")

    known_docs = set(available_doc_ids) if available_doc_ids is not None else None
    seen_ids = set()
    allowed_keys = {"id", "question", "relevant_docs"}
    for index, item in enumerate(questions):
        location = f"{dataset_path}: record {index + 1}"
        if not isinstance(item, dict):
            raise DatasetValidationError(f"{location} must be an object")
        unknown = sorted(set(item) - allowed_keys)
        if unknown:
            raise DatasetValidationError(f"{location} has unknown field(s): {', '.join(unknown)}")

        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise DatasetValidationError(f"{location} requires a nonempty string id")
        if question_id in seen_ids:
            raise DatasetValidationError(f"{location} duplicates question id {question_id!r}")
        seen_ids.add(question_id)
        location = f"{dataset_path}: question {question_id!r}"

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DatasetValidationError(f"{location} requires a nonempty string question")
        relevant_docs = item.get("relevant_docs")
        if not isinstance(relevant_docs, list) or not relevant_docs:
            raise DatasetValidationError(f"{location} requires a nonempty relevant_docs array")
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in relevant_docs):
            raise DatasetValidationError(f"{location} relevant_docs values must be nonempty strings")
        if len(set(relevant_docs)) != len(relevant_docs):
            raise DatasetValidationError(f"{location} relevant_docs must not contain duplicates")
        if known_docs is not None:
            missing = sorted(set(relevant_docs) - known_docs)
            if missing:
                raise DatasetValidationError(
                    f"{location} references unknown document id(s): {', '.join(missing)}"
                )
    return questions


def evaluate(cfg: Optional[Config] = None) -> Tuple[Dict[str, float], List[dict]]:
    cfg = cfg or Config.from_env()
    if isinstance(cfg.top_k, bool) or not isinstance(cfg.top_k, int) or cfg.top_k <= 0:
        raise DatasetValidationError("top_k must be a positive integer")

    documents = load_documents(cfg.docs_dir)
    chunks = build_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)
    retriever = TfidfRetriever().fit(chunks)

    questions = load_eval(cfg.eval_path, (doc_id for doc_id, _ in documents))
    audit = AuditLog(cfg.logs_dir / "audit.jsonl", cfg.log_query_text)
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
    try:
        metrics, per_question = evaluate(cfg)
    except DatasetValidationError as error:
        print(f"rag-evaluate: {error}", file=sys.stderr)
        raise SystemExit(2) from None

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
