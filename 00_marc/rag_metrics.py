# rag_metrics.py
# Utility functions to evaluate a Retrieval‑Augmented Generation (RAG) pipeline.
# Supports both reference‑free metrics (grounding, overlap, retrieval relevance)
# and reference‑based metrics (EM, token‑F1, ROUGE-L) when labels are available.

"""
Métricas reference-free
No requieren respuesta de referencia; evalúan únicamente la relación entre la respuesta generada y 
los contextos recuperados o la pregunta.

- grounding_score: Qué mide: la proporción de tokens de la respuesta que aparecen en cualquiera de los 
textos de contexto.

Interpretación: entre 0 y 1; un valor alto (cercano a 1) indica que casi todas las palabras de la 
respuesta están “fundamentadas” en el material recuperado, reduciendo el riesgo de invención 
(alucinaciones).

- context_overlap_f1: Qué mide: la F1 entre los tokens de la respuesta y los tokens de los textos de
contexto recuperados.

Interpretación:  alto si la respuesta usa bien la información disponible (precisión) y al mismo 
tiempo refleja buena parte de lo recuperado (cobertura).

- question_context_similarity: F1 token-level entre la pregunta y la concatenación de los contextos 
recuperados.

Interpretación: refleja qué tan relevantes son realmente los fragmentos recuperados para responder 
al usuario. Un valor cercano a 1 significa que gran parte del texto de contexto comparte vocabulario 
clave con la pregunta.

Métricas reference-based

Requieren que cada registro lleve un campo "reference" con la respuesta correcta conocida.

- exact_match_score (EM): coincidencia exacta entre la respuesta generada y la referencia, 
tras normalizar (minusculas, quitar puntuación).

Interpretación: binaria (0 o 1). Es la métrica más estricta: solo acepta respuestas textuales 
idénticas.

- token_f1_score:  F1 entre los tokens de la predicción y los de la referencia, cuantificando 
solapamiento parcial.

Interpretación: útil cuando la respuesta puede expresarse con palabras distintas pero comparte 
la misma información (p. ej. distinto orden, sinónimos).

- rouge_l_score: ROUGE-L basado en la longest common subsequence (LCS) de tokens entre predicción
y referencia.

Interpretación: capta solapamientos de secuencias (no solo multiconjunto). Un LCS largo implica 
que la predicción respeta gran parte del orden y la estructura de la referencia.
"""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import Iterable, List, Mapping, Sequence

###############################################################################
# Text normalization and tokenization
###############################################################################

def _normalize(text: str) -> str:
    """Lower‑case, strip punctuation and collapse whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Split normalized text on whitespace."""
    return _normalize(text).split()

###############################################################################
# Reference‑free metrics (no ground truth answer required)
###############################################################################

def grounding_score(answer: str, contexts: Iterable[str]) -> float:
    """Precision: proportion of answer tokens present in retrieved contexts."""
    ans = _tokenize(answer)
    if not ans:
        return 0.0
    ctx = set(_tokenize(" ".join(contexts)))
    supported = sum(1 for t in ans if t in ctx)
    return supported / len(ans)


def context_overlap_f1(answer: str, contexts: Iterable[str]) -> float:
    """F1 between answer tokens and context tokens (utilisation)."""
    ans = _tokenize(answer)
    ctx = _tokenize(" ".join(contexts))
    if not ans or not ctx:
        return 0.0
    common = Counter(ans) & Counter(ctx)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(ans)
    r = n_same / len(ctx)
    return 2 * p * r / (p + r)


def question_context_similarity(question: str, contexts: Iterable[str]) -> float:
    """F1 between question and retrieved contexts (relevance)."""
    q = _tokenize(question)
    ctx = _tokenize(" ".join(contexts))
    if not q or not ctx:
        return 0.0
    common = Counter(q) & Counter(ctx)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(q)
    r = n_same / len(ctx)
    return 2 * p * r / (p + r)

###############################################################################
# Reference‑based metrics (requires labeled ground truth answer)
###############################################################################

def exact_match_score(prediction: str, reference: str) -> float:
    """Binary exact match after normalization."""
    return float(_normalize(prediction) == _normalize(reference))


def token_f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pred_tokens)
    r = n_same / len(ref_tokens)
    return 2 * p * r / (p + r)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute length of longest common subsequence between two token lists."""
    # dynamic programming
    m, n = len(a), len(b)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score based on LCS."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

###############################################################################
# Dataset evaluation
###############################################################################

def evaluate_dataset(records: Iterable[Mapping[str, object]]) -> Mapping[str, float]:
    """Aggregate RAG reference-free metrics across records."""
    gs, f1s, sims = [], [], []
    for rec in records:
        a = str(rec["respuesta"])
        ctx = rec["contextos"]
        q = str(rec["pregunta"])
        gs.append(grounding_score(a, ctx))
        f1s.append(context_overlap_f1(a, ctx))
        sims.append(question_context_similarity(q, ctx))
    n = max(len(gs), 1)
    return {
        "GroundingScore": sum(gs)/n,
        "ContextOverlapF1": sum(f1s)/n,
        "QuestionContextSim": sum(sims)/n
    }


def evaluate_with_references(records: Iterable[Mapping[str, object]]) -> Mapping[str, float]:
    """Aggregate both reference-based and reference-free metrics.  
    Records must include keys:
      - "respuesta" (prediction)
      - "reference" (ground truth answer)
      - "pregunta", "contextos"
    """
    ems, f1_r, rouge_l, gs, f1s, sims = [], [], [], [], [], []
    for rec in records:
        pred = str(rec["respuesta"])
        ref  = str(rec.get("reference", ""))
        ctx  = rec.get("contextos", [])
        q    = str(rec.get("pregunta", ""))
        ems.append(exact_match_score(pred, ref))
        f1_r.append(token_f1_score(pred, ref))
        rouge_l.append(rouge_l_score(pred, ref))
        gs.append(grounding_score(pred, ctx))
        f1s.append(context_overlap_f1(pred, ctx))
        sims.append(question_context_similarity(q, ctx))
    n = max(len(ems), 1)
    return {
        "ExactMatch": sum(ems)/n,
        "TokenF1":    sum(f1_r)/n,
        "ROUGE_L":    sum(rouge_l)/n,
        "GroundingScore":  sum(gs)/n,
        "ContextOverlapF1":  sum(f1s)/n,
        "QuestionContextSim": sum(sims)/n
    }