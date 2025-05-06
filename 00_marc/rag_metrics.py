# rag_metrics.py
"""Utility functions to evaluate a Retrieval‑Augmented Generation (RAG) pipeline **when you do NOT have a ground‑truth (referencia) answer**.

Each record produced by your system must contain at least these keys::

    {
        "pregunta": "...",          # The user question
        "respuesta": "...",        # The model answer
        "contextos": ["...", ...]  # Ranked list of retrieved context strings
    }

Provided metrics (all range 0‑1)
--------------------------------
1. **grounding_score** – Precision‑style metric: proportion of *answer* tokens that appear in any retrieved context, i.e. how well the answer is supported by the evidence.
2. **context_overlap_f1** – Symmetric F1 between answer tokens and context tokens (harmonic mean of *grounding precision* and *context coverage recall*). High only if the answer both draws from and covers the retrieved information.
3. **question_context_similarity** – Token‑level F1 between the *question* and the aggregated *contexts*; measures how relevant the retrieved contexts are for the user query.

Example::

    >>> record = {
    ...     "pregunta": "¿Capital de Francia?",
    ...     "respuesta": "París es la capital francesa.",
    ...     "contextos": ["París es la capital y ciudad más grande de Francia."]
    ... }
    >>> import rag_metrics as rm
    >>> rm.grounding_score(record["respuesta"], record["contextos"])
    1.0
    >>> rm.context_overlap_f1(record["respuesta"], record["contextos"])
    0.8
    >>> rm.question_context_similarity(record["pregunta"], record["contextos"])
    0.6

You can aggregate these over a dataset with **evaluate_dataset**.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Mapping

###############################################################################
# Text normalisation helpers
###############################################################################

def _normalize(text: str) -> str:
    """Lower‑case, strip punctuation/extra whitespace and normalise accents."""
    text = text.lower()
    # Replace punctuation with spaces so words stay separated
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Normalise and split on whitespace."""
    return _normalize(text).split()

###############################################################################
# Metric 1 – Grounding Score (precision)
###############################################################################

def grounding_score(answer: str, contexts: Iterable[str]) -> float:
    """Fraction of *answer* tokens that appear in the concatenation of *contexts*."""
    ans_toks = _tokenize(answer)
    if not ans_toks:
        return 0.0
    ctx_tokens = set(_tokenize(" ".join(contexts)))
    supported = sum(1 for tok in ans_toks if tok in ctx_tokens)
    return supported / len(ans_toks)

###############################################################################
# Metric 2 – Context‑Overlap F1 (utilisation)
###############################################################################

def context_overlap_f1(answer: str, contexts: Iterable[str]) -> float:
    """Harmonic mean of (i) how much of the answer is in contexts (precision) and
    (ii) how much of the contexts is used in the answer (recall)."""
    ans_toks = _tokenize(answer)
    ctx_toks = _tokenize(" ".join(contexts))
    if not ans_toks or not ctx_toks:
        return 0.0
    common = Counter(ans_toks) & Counter(ctx_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(ans_toks)      # == grounding_score
    recall = num_same / len(ctx_toks)         # context coverage
    return 2 * precision * recall / (precision + recall)

###############################################################################
# Metric 3 – Question‑Context Similarity (retrieval relevance)
###############################################################################

def question_context_similarity(question: str, contexts: Iterable[str]) -> float:
    """Token‑level F1 between the *question* and the concatenated *contexts*."""
    q_toks = _tokenize(question)
    ctx_toks = _tokenize(" ".join(contexts))
    if not q_toks or not ctx_toks:
        return 0.0
    common = Counter(q_toks) & Counter(ctx_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(q_toks)
    recall = num_same / len(ctx_toks)
    return 2 * precision * recall / (precision + recall)

###############################################################################
# Convenience – Evaluate an entire dataset
###############################################################################

def evaluate_dataset(records: Iterable[Mapping[str, object]]) -> Mapping[str, float]:
    """Aggregate the three metrics over an iterable of RAG records."""
    g_scores, f1s, qc_sims = [], [], []
    for rec in records:
        answer = str(rec["respuesta"])
        contexts = rec["contextos"]
        question = str(rec["pregunta"])
        g_scores.append(grounding_score(answer, contexts))
        f1s.append(context_overlap_f1(answer, contexts))
        qc_sims.append(question_context_similarity(question, contexts))
    n = max(len(g_scores), 1)
    return {
        "Grounding": sum(g_scores) / n,
        "ContextOverlapF1": sum(f1s) / n,
        "QuestionContextSim": sum(qc_sims) / n,
    }

###############################################################################
# CLI entry‑point (optional) – evaluate a JSONL file
###############################################################################

if __name__ == "__main__":
    import json, sys, pathlib, argparse

    p = argparse.ArgumentParser(description="Evaluate RAG outputs (no reference answer required).")
    p.add_argument("file", type=pathlib.Path, help="Path to JSONL with RAG records (one per line)")
    args = p.parse_args()
    with args.file.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    metrics = evaluate_dataset(records)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
