"""Privacy and validation guarantees for local audit logging."""
import hashlib
import importlib
import json

import pytest

from rag_lab.audit import AuditLog
from rag_lab.config import Config, ConfigError, _as_bool
from rag_lab.evaluate import evaluate
from rag_lab.retriever import RetrievalResult


def test_audit_log_is_hash_only_by_default_and_never_logs_chunk_text(tmp_path):
    query = "מהי מדיניות החופשה?"
    corpus_text = "private synthetic corpus sentence"
    result = RetrievalResult("doc#0", "doc", 0.75, corpus_text)
    path = tmp_path / "audit.jsonl"

    entry = AuditLog(path).log_query(query, [result])
    serialized = path.read_text(encoding="utf-8")
    assert "query" not in entry
    assert entry["query_sha256"] == hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert query not in serialized
    assert corpus_text not in serialized
    assert json.loads(serialized)["retrieved"] == [
        {"chunk_id": "doc#0", "doc_id": "doc", "score": 0.75}
    ]


def test_audit_log_raw_query_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "audit.jsonl"
    entry = AuditLog(path, log_query_text=True).log_query("explicit synthetic query", [])
    assert entry["query"] == "explicit synthetic query"
    assert json.loads(path.read_text(encoding="utf-8"))["query"] == "explicit synthetic query"


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_boolean_parser_accepts_explicit_true_values(value):
    assert _as_bool(value) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_boolean_parser_accepts_explicit_false_values(value):
    assert _as_bool(value) is False


@pytest.mark.parametrize("value", ["", "maybe", "enabled", "2"])
def test_boolean_parser_rejects_ambiguous_values(value):
    with pytest.raises(ConfigError, match="boolean value must be one of"):
        _as_bool(value)


def test_config_from_env_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "maybe")
    with pytest.raises(ConfigError, match="boolean value must be one of"):
        Config.from_env()


def test_config_from_env_requires_explicit_true_opt_in(monkeypatch):
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "true")
    assert Config.from_env().log_query_text is True


def test_default_evaluation_log_omits_query_and_corpus_text(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    corpus_text = "synthetic confidential phrase unique-token"
    (docs_dir / "doc.md").write_text(corpus_text, encoding="utf-8")
    eval_path = tmp_path / "questions.json"
    query = "Where is unique-token?"
    eval_path.write_text(json.dumps([
        {"id": "q1", "question": query, "relevant_docs": ["doc"]}
    ]), encoding="utf-8")
    cfg = Config(docs_dir=docs_dir, eval_path=eval_path, logs_dir=tmp_path / "logs")

    evaluate(cfg)
    serialized = (cfg.logs_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert query not in serialized
    assert corpus_text not in serialized


def test_evaluate_cli_reports_invalid_boolean_without_traceback(monkeypatch, capsys):
    evaluate_module = importlib.import_module("rag_lab.evaluate")

    def invalid_config():
        raise ConfigError("synthetic invalid boolean")

    monkeypatch.setattr(evaluate_module.Config, "from_env", invalid_config)
    with pytest.raises(SystemExit) as exit_info:
        evaluate_module.main()
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert captured.err == "rag-evaluate: synthetic invalid boolean\n"
    assert "Traceback" not in captured.err
