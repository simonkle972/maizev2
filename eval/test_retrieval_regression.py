"""Pytest regression test for retrieval.

Parametrized over eval/maize_eval_v1.jsonl. Fails on regressions to in-corpus
non-not_in_corpus rows whose `failure_type_target` is null (i.e., working
cases). Type A-E rows are not asserted as passing here — they're failing today
by construction; the harness reports them via run_eval.py's scorecard.

After the implementation gate, this file should be EXTENDED with assertions
for the Type A-E rows once the refined retriever can pass them.

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 pytest eval/test_retrieval_regression.py -v
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

EVAL_FILE = Path(__file__).parent / "maize_eval_v1.jsonl"


def load_rows():
    with EVAL_FILE.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def working_case_rows():
    return [r for r in load_rows()
            if r["failure_type_target"] is None and not r.get("not_in_corpus")]


@pytest.fixture(scope="session")
def app_ctx():
    from app import app
    ctx = app.app_context()
    ctx.push()
    yield app
    ctx.pop()


@pytest.fixture(scope="session")
def retrieve_context_fn():
    from src.retriever import retrieve_context
    return retrieve_context


@pytest.mark.parametrize("row", working_case_rows(), ids=lambda r: r["row_id"])
def test_working_case_must_not_regress(row, app_ctx, retrieve_context_fn):
    """Working-case rows: retrieval must include at least one correct doc in top-5
    AND must not return any forbidden doc."""
    session_id = f"pytest-{uuid.uuid4()}"
    chunks, _diag = retrieve_context_fn(
        ta_id=row["ta_id"],
        query=row["query"],
        top_k=8,
        conversation_history=list(row.get("prior_turns") or []),
        session_id=session_id,
        course_name="Introduction to Data Analysis and Econometrics",
    )
    retrieved = [c.get("file_name", "") for c in (chunks or [])]
    top5 = set(retrieved[:5])
    correct = set(row["correct_doc_ids"])
    forbidden = set(row.get("forbidden_doc_ids") or [])

    assert top5 & correct, (
        f"working-case row {row['row_id']}: none of correct_doc_ids "
        f"{sorted(correct)} appeared in top-5 retrieved {sorted(top5)}"
    )
    assert not (set(retrieved) & forbidden), (
        f"working-case row {row['row_id']}: forbidden doc(s) appeared in retrieved "
        f"{sorted(set(retrieved) & forbidden)}"
    )
