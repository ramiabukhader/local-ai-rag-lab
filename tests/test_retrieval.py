"""Smoke tests for the retriever and evaluation harness."""
from rag_lab.config import Config
from rag_lab.evaluate import evaluate
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
