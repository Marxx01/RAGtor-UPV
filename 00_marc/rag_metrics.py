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

- cosine_similarity: Similaridad entre la representación vectorial de la respuesta y la referencia,
  usando el modelo de lenguaje para calcular la distancia entre ambos textos.

Interpretación: entre 0 y 1; un valor alto indica que la respuesta generada es semánticamente
similar a la referencia. Útil para evaluar respuestas que pueden no coincidir palabra por palabra,
pero que son semánticamente equivalentes.

- avg_l2_distance: Distancia L2 promedio entre la representación vectorial de la respuesta y la
  referencia, usando el modelo de lenguaje para calcular la distancia entre ambos textos.
  Interpretación: entre 0 y 1; un valor bajo indica que la respuesta generada es semánticamente
  similar a la referencia. Útil para evaluar respuestas que pueden no coincidir palabra por palabra,
  pero que son semánticamente equivalentes.




Interpretación: capta solapamientos de secuencias (no solo multiconjunto). Un LCS largo implica 
que la predicción respeta gran parte del orden y la estructura de la referencia.
"""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import Iterable, List, Mapping, Sequence
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

###############################################################################
# Text normalization and tokenization
###############################################################################

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Split normalized text on whitespace."""
    return _normalize(text).split()

def _tokenize_llm(text: str, tokenizer_name: str) -> List[str]:
    """Tokenize using LLM tokenizer using sentence transformer model from huggingface"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(tokenizer_name, *{"device": "cuda" if torch.cuda.is_available() else "cpu"})
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

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

def cosine_similarity_score(prediction: str, reference: str, model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> float:
    """Compute cosine similarity between prediction and reference using LLM embeddings."""
    pred_tokens = _tokenize_llm(prediction, model)
    ref_tokens = _tokenize_llm(reference, model)
    if not pred_tokens.size or not ref_tokens.size:
        return 0.0
    pred_vector = np.array(pred_tokens).reshape(1, -1)
    ref_vector = np.array(ref_tokens).reshape(1, -1)
    similarity = float(cosine_similarity(pred_vector, ref_vector)[0][0])
    return similarity

def cosine_similarity_score_context(prediction: str, context: List[str], model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> float:
    """Compute cosine similarity between prediction and context using LLM embeddings."""
    pred_tokens = _tokenize_llm(prediction, model)
    ctx_tokens = [_tokenize_llm(ctx, model) for ctx in context]
    if not pred_tokens.size or not ctx_tokens:
        return 0.0
    pred_vector = np.array(pred_tokens).reshape(1, -1)
    ctx_vectors = np.array(ctx_tokens).reshape(len(ctx_tokens), -1)
    similarities = cosine_similarity(pred_vector, ctx_vectors)
    return float(similarities.mean())

def avg_l2_distance(prediction: str, reference: str, model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> float:
    """Compute average L2 distance between prediction and reference using LLM embeddings."""
    pred_tokens = _tokenize_llm(prediction, model)
    ref_tokens = _tokenize_llm(reference, model)
    if not pred_tokens.size or not ref_tokens.size:
        return 0.0
    pred_vector = np.array(pred_tokens).reshape(1, -1)
    ref_vector = np.array(ref_tokens).reshape(1, -1)
    distance = np.linalg.norm(pred_vector - ref_vector)
    return float(distance.mean())

def avg_l2_distance_context(prediction: str, context: List[str], model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> float:
    """Compute average L2 distance between prediction and context using LLM embeddings."""
    pred_tokens = _tokenize_llm(prediction, model)
    ctx_tokens = [_tokenize_llm(ctx, model) for ctx in context]
    if not pred_tokens.size or not ctx_tokens:
        return 0.0
    pred_vector = np.array(pred_tokens).reshape(1, -1)
    ctx_vectors = np.array(ctx_tokens).reshape(len(ctx_tokens), -1)
    distances = np.linalg.norm(pred_vector - ctx_vectors, axis=1)
    return float(distances.mean())
