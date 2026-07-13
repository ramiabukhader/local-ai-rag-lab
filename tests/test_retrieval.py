"""Tests for the retriever and evaluation harness."""
import importlib
import json

import pytest

from rag_lab.config import Config
from rag_lab.evaluate import DatasetValidationError, evaluate, load_eval
from rag_lab.ingest import build_chunks
from rag_lab.retriever import TfidfRetriever


def test_retriever_ranks_relevant_doc_first():
    docs = [
        ("leave-policy", "Full-time employees get 25 days of annual leave per year."),
        ("expense-policy", "The daily meal expense limit is 40 units and receipts are required."),
    ]
    retriever = TfidfRetriever().fit(build_chunks(docs, chunk_size=40, overlap=10))
    top = retriever.query("How many days of annual leave?", top_k=1)
    assert top and top[0].doc_id == "leave-policy"


def test_evaluation_meets_baseline_on_samples():
    metrics, _ = evaluate(Config())
    assert metrics["questions"] == 6
    # The sample set is designed so every question's relevant doc is retrievable.
    assert metrics[f"recall@{Config().top_k}"] == 1.0


def test_load_eval_accepts_multiple_relevant_documents(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "multi", "question": "Which policies apply?", "relevant_docs": ["a", "b"]}
    ]), encoding="utf-8")
    questions = load_eval(path, {"a", "b"})
    assert questions[0]["relevant_docs"] == ["a", "b"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "invalid JSON"),
        ({}, "top-level value must be an array"),
        ([], "at least one question"),
        (["bad"], "record 1 must be an object"),
        ([{"question": "q", "relevant_docs": ["doc"]}], "nonempty string id"),
        ([{"id": "q1", "question": " ", "relevant_docs": ["doc"]}], "nonempty string question"),
        ([{"id": "q1", "question": "q", "relevant_docs": []}], "nonempty relevant_docs"),
        ([{"id": "q1", "question": "q", "relevant_docs": [3]}], "nonempty strings"),
        ([{"id": "q1", "question": "q", "relevant_docs": ["doc", "doc"]}], "must not contain duplicates"),
        ([{"id": "q1", "question": "q", "relevant_docs": ["missing"]}], "unknown document"),
        ([
            {"id": "q1", "question": "first", "relevant_docs": ["doc"]},
            {"id": "q1", "question": "second", "relevant_docs": ["doc"]},
        ], "duplicates question id"),
        ([{"id": "q1", "question": "q", "relevant_docs": ["doc"], "relevent_docs": []}], "unknown field"),
    ],
)
def test_load_eval_rejects_invalid_datasets(tmp_path, payload, message):
    path = tmp_path / "questions.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        load_eval(path, {"doc"})


def test_load_eval_reports_missing_file_without_question_content(tmp_path):
    with pytest.raises(DatasetValidationError, match="cannot read evaluation dataset"):
        load_eval(tmp_path / "missing.json", {"doc"})


def test_evaluate_validation_failure_does_not_create_audit_log(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "known.md").write_text("known content", encoding="utf-8")
    eval_path = tmp_path / "bad.json"
    eval_path.write_text(json.dumps([
        {"id": "bad", "question": "synthetic question", "relevant_docs": ["unknown"]}
    ]), encoding="utf-8")
    logs_dir = tmp_path / "logs"
    cfg = Config(docs_dir=docs_dir, eval_path=eval_path, logs_dir=logs_dir)

    with pytest.raises(DatasetValidationError, match="unknown document"):
        evaluate(cfg)
    assert not logs_dir.exists()


def test_evaluate_cli_reports_validation_errors_without_traceback(monkeypatch, capsys):
    evaluate_module = importlib.import_module("rag_lab.evaluate")

    def fail_validation(_cfg):
        raise DatasetValidationError("synthetic dataset failure")

    monkeypatch.setattr(evaluate_module, "evaluate", fail_validation)
    with pytest.raises(SystemExit) as exit_info:
        evaluate_module.main()
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert captured.err == "rag-evaluate: synthetic dataset failure\n"
    assert "Traceback" not in captured.err
