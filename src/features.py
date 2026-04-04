"""
src/features.py
---------------
12-Dimensional QPP Feature Extraction for RAG-QPP.

Features are grouped into three categories:
  (A) Semantic / similarity-based  : max_sim, sim_variance, high_sim_count,
                                     rank_dropoff, emb_variance
  (B) Lexical / term-based         : term_overlap, query_length,
                                     query_idf_sum, query_entropy
  (C) Classical QPP indicators     : clarity, wig, nqc

All features are computed entirely from the post-retrieval output
(query + top-k passages + their embeddings).  No relevance judgments
or generator signals are required.

Reference: Sinha & Chakma (2026), Section 3.13
"""

from __future__ import annotations

import math
import logging
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HIGH_SIM_THRESHOLD = 0.8        # threshold for high_sim_count  (Eq. 22)
NQC_CLIP           = 10.0       # upper bound for NQC            (Eq. 31)
EPSILON            = 1e-5       # smoothing constant
TOP_K_CLASSIC_QPP  = 5          # Clarity / WIG computed on top-5 passages
CORPUS_SIZE_DEFAULT = 8_841_823  # MS MARCO Passage corpus size


# ══════════════════════════════════════════════════════════════════════════════
# Cosine similarity helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPSILON
    return float(np.dot(a, b) / denom)


def _cosine_batch(query_emb: np.ndarray,
                  passage_embs: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity: query (d,) vs each row of passage_embs (k, d).
    Returns shape (k,).
    """
    q_norm = query_emb / (np.linalg.norm(query_emb) + EPSILON)
    p_norms = passage_embs / (
        np.linalg.norm(passage_embs, axis=1, keepdims=True) + EPSILON
    )
    return p_norms @ q_norm  # (k,)


# ══════════════════════════════════════════════════════════════════════════════
# (A) Semantic / similarity-based features  (Eqs. 20-24)
# ══════════════════════════════════════════════════════════════════════════════

def compute_max_sim(sims: np.ndarray) -> float:
    """
    Feature 1 — max_sim (Eq. 20):
    Highest cosine similarity between query and any retrieved passage.
    High value → at least one passage is strongly aligned with the query.
    """
    return float(np.max(sims))


def compute_sim_variance(sims: np.ndarray) -> float:
    """
    Feature 2 — sim_variance (Eq. 21):
    Variance of cosine similarity scores across all top-k passages.
    Moderate variance → mixture of relevant and less-relevant passages.
    """
    return float(np.var(sims))


def compute_high_sim_count(sims: np.ndarray,
                           threshold: float = HIGH_SIM_THRESHOLD) -> float:
    """
    Feature 3 — high_sim_count (Eq. 22):
    Number of passages whose similarity exceeds `threshold` (default 0.8).
    Higher count → stronger evidence pool for the generator.
    """
    return float(np.sum(sims > threshold))


def compute_rank_dropoff(sims: np.ndarray) -> float:
    """
    Feature 4 — rank_dropoff (Eq. 23):
    Difference in cosine similarity between the 1st and 10th ranked passage.
    Large drop → only the very top passage is relevant.
    Small drop → consistent relevance across top ranks.
    """
    if len(sims) < 10:
        return float(sims[0] - sims[-1])
    return float(sims[0] - sims[9])


def compute_emb_variance(passage_embs: np.ndarray) -> float:
    """
    Feature 5 — emb_variance (Eq. 24):
    Average squared distance of each passage embedding from the centroid.
    High value → retrieved passages are semantically scattered (hard query).
    Low value  → tight coherent cluster (easy query).

    This is the single most informative feature in the ablation study.
    """
    centroid = passage_embs.mean(axis=0)                    # (d,)
    diffs    = passage_embs - centroid                       # (k, d)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


# ══════════════════════════════════════════════════════════════════════════════
# (B) Lexical / term-based features  (Eqs. 25-28)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lower-case tokenizer."""
    return text.lower().split()


def compute_term_overlap(query: str, passages: List[str]) -> float:
    """
    Feature 6 — term_overlap (Eq. 25):
    Average fraction of query terms found in each retrieved passage.
    Higher → stronger lexical alignment between query and evidence.
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    overlaps = []
    for p in passages:
        p_terms = set(_tokenize(p))
        overlap = len(query_terms & p_terms) / len(query_terms)
        overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


def compute_query_length(query: str) -> float:
    """
    Feature 7 — query_length (Eq. 26):
    Number of tokens in the query after whitespace tokenization.
    Short queries → often vague; long queries → more specific context.
    """
    return float(len(_tokenize(query)))


def compute_query_idf_sum(query: str,
                           df_lookup: Dict[str, int],
                           corpus_size: int = CORPUS_SIZE_DEFAULT) -> float:
    """
    Feature 8 — query_idf_sum (Eq. 27):
    Sum of IDF scores for each query term.
    idf(t) = log(N / (n_t + 1))
    High sum → query contains rare, discriminative terms.

    Args:
        df_lookup  : {term: document_frequency} mapping for the corpus.
        corpus_size: total number of documents N in the corpus.
    """
    terms = _tokenize(query)
    total = 0.0
    for t in terms:
        n_t  = df_lookup.get(t, 0)
        total += math.log(corpus_size / (n_t + 1))
    return total


def compute_query_entropy(query: str,
                           df_lookup: Dict[str, int],
                           corpus_size: int = CORPUS_SIZE_DEFAULT) -> float:
    """
    Feature 9 — query_entropy (Eq. 28):
    Shannon entropy of the IDF-weighted query term distribution.
    Higher entropy → more even distribution of informativeness among terms.
    """
    terms = _tokenize(query)
    if not terms:
        return 0.0
    idfs = np.array([
        math.log(corpus_size / (df_lookup.get(t, 0) + 1))
        for t in terms
    ], dtype=float)
    total = idfs.sum() + EPSILON
    probs = idfs / total
    # Shannon entropy: -Σ p log p  (skip zero entries)
    entropy = -np.sum(probs * np.log(probs + EPSILON))
    return float(entropy)


# ══════════════════════════════════════════════════════════════════════════════
# (C) Classical QPP indicators  (Eqs. 29-31)
# ══════════════════════════════════════════════════════════════════════════════

def compute_clarity(passages: List[str],
                    corpus_lm: Dict[str, float]) -> float:
    """
    Feature 10 — Clarity (Eq. 29):
    KL-divergence between top-5 passage language model and corpus LM.
    High clarity → retrieved passages are topically coherent relative to corpus.

    Args:
        passages  : top-5 retrieved passage texts.
        corpus_lm : {word: P(w|C)} unigram corpus language model.
    """
    passages = passages[:TOP_K_CLASSIC_QPP]

    # Build retrieved-set language model (unigram, add-1 smoothed)
    word_counts: Dict[str, int] = {}
    total_tokens = 0
    for p in passages:
        for w in _tokenize(p):
            word_counts[w] = word_counts.get(w, 0) + 1
            total_tokens   += 1

    if total_tokens == 0:
        return 0.0

    clarity = 0.0
    for w, cnt in word_counts.items():
        p_w_r = cnt / total_tokens
        p_w_c = corpus_lm.get(w, EPSILON)   # fall back to epsilon if OOV
        if p_w_r > 0 and p_w_c > 0:
            clarity += p_w_r * math.log(p_w_r / p_w_c)
    return float(clarity)


def compute_wig(sims: np.ndarray) -> float:
    """
    Feature 11 — WIG — Weighted Information Gain (Eq. 30):
    Average similarity gain of the top-5 passages over the 6th passage.
    High WIG → clear separation between relevant and marginal results.

    Uses the same top-5 window as the generator.
    """
    if len(sims) < 6:
        return float(np.mean(sims[:5]) - sims[-1]) if len(sims) > 1 else 0.0

    top5_mean = float(np.mean(sims[:5]))
    ref_sim   = float(sims[5])          # 6th-ranked passage as reference
    return top5_mean - ref_sim


def compute_nqc(sims: np.ndarray) -> float:
    """
    Feature 12 — NQC — Normalized Query Commitment (Eq. 31):
    Coefficient of variation of cosine similarities over top-20 passages,
    clipped at NQC_CLIP=10 to suppress extreme outliers.

    High NQC → retriever is internally confident (clear score separation).

    NOTE: NQC is computed over the full top-20, unlike Clarity/WIG (top-5).
    """
    mu    = float(np.mean(sims)) + EPSILON
    sigma = float(np.std(sims))
    return float(min(sigma / mu, NQC_CLIP))


# ══════════════════════════════════════════════════════════════════════════════
# Main feature-extraction function
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(
    query: str,
    passage_texts: List[str],
    passage_embs: np.ndarray,
    query_emb: np.ndarray,
    df_lookup: Dict[str, int],
    corpus_lm: Dict[str, float],
    corpus_size: int = CORPUS_SIZE_DEFAULT,
) -> np.ndarray:
    """
    Extract the full 12-dimensional QPP feature vector for a single query.

    Parameters
    ----------
    query         : raw query string.
    passage_texts : texts of the top-k retrieved passages (k ≤ 20).
    passage_embs  : numpy array of shape (k, 768) — passage embeddings.
    query_emb     : numpy array of shape (768,)    — query embedding.
    df_lookup     : {term: doc_freq} for IDF computation.
    corpus_lm     : {word: P(w|C)} corpus unigram language model.
    corpus_size   : total number of documents in the corpus.

    Returns
    -------
    np.ndarray of shape (12,), dtype float32.
    """
    sims = _cosine_batch(query_emb, passage_embs)   # (k,)

    features = np.array([
        # ── (A) Semantic ───────────────────────────────────────────
        compute_max_sim(sims),                                      # 1
        compute_sim_variance(sims),                                 # 2
        compute_high_sim_count(sims),                               # 3
        compute_rank_dropoff(sims),                                 # 4
        compute_emb_variance(passage_embs),                        # 5
        # ── (B) Lexical ────────────────────────────────────────────
        compute_term_overlap(query, passage_texts),                 # 6
        compute_query_length(query),                                # 7
        compute_query_idf_sum(query, df_lookup, corpus_size),       # 8
        compute_query_entropy(query, df_lookup, corpus_size),       # 9
        # ── (C) Classical QPP ──────────────────────────────────────
        compute_clarity(passage_texts, corpus_lm),                  # 10
        compute_wig(sims),                                          # 11
        compute_nqc(sims),                                          # 12
    ], dtype=np.float32)

    return features


FEATURE_NAMES = [
    "max_sim",        # 1
    "sim_variance",   # 2
    "high_sim_count", # 3
    "rank_dropoff",   # 4
    "emb_variance",   # 5
    "term_overlap",   # 6
    "query_length",   # 7
    "query_idf_sum",  # 8
    "query_entropy",  # 9
    "clarity",        # 10
    "wig",            # 11
    "nqc",            # 12
]


# ══════════════════════════════════════════════════════════════════════════════
# Batch extraction + normalization
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_batch(
    queries: List[str],
    passages_list: List[List[str]],
    passage_embs_list: List[np.ndarray],
    query_embs: np.ndarray,
    df_lookup: Dict[str, int],
    corpus_lm: Dict[str, float],
    corpus_size: int = CORPUS_SIZE_DEFAULT,
) -> np.ndarray:
    """
    Extract features for a batch of queries.

    Returns
    -------
    np.ndarray of shape (N, 12).
    """
    all_features = []
    for i, (query, passages, pembs, qemb) in enumerate(
        zip(queries, passages_list, passage_embs_list, query_embs)
    ):
        try:
            fv = extract_features(
                query, passages, pembs, qemb,
                df_lookup, corpus_lm, corpus_size
            )
        except Exception as exc:
            logger.warning("Feature extraction failed for query %d: %s", i, exc)
            fv = np.zeros(12, dtype=np.float32)
        all_features.append(fv)

    return np.stack(all_features, axis=0)   # (N, 12)


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """
    Z-score normalize features using statistics from X_train.
    Follows the paper's use of sklearn StandardScaler (Section 3.13).

    Returns
    -------
    X_train_norm, X_test_norm (or None), fitted_scaler
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test) if X_test is not None else None
    return X_train_norm, X_test_norm, scaler
