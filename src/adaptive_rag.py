"""
src/adaptive_rag.py
-------------------
QPP-Guided Adaptive Retrieval-Augmented Generation.

Implements the three-level adaptive strategy from Section 3.7 / Section 4:

  Low-QPP  (ŷ < τ₁)       → k = 50   (deep retrieval)
  Medium   (τ₁ ≤ ŷ < τ₂)  → k = 30
  High-QPP (ŷ ≥ τ₂)       → k = 20   (standard retrieval)

where τ₁ = 33rd percentile and τ₂ = 66th percentile of predicted scores.

The QPP score is derived exclusively from post-retrieval features;
no generator-internal signals (perplexity, entropy) are used.

Also implements the learning-free query-length heuristic baseline used
in Section 6 for comparison.

Reference: Sinha & Chakma (2026), Sections 3.7, 4, 5.5, 6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.retrieval import BaseRetriever
from src.generation import BaseGenerator
from src.features import extract_features, normalize_features
from src.evaluate import compute_generation_metrics, stratified_analysis

logger = logging.getLogger(__name__)

# ── Adaptive depth table  (Eq. 46) ───────────────────────────────────────────
K_LOW    = 50
K_MEDIUM = 30
K_HIGH   = 20


# ══════════════════════════════════════════════════════════════════════════════
# QPP-based adaptive depth selector
# ══════════════════════════════════════════════════════════════════════════════

def adaptive_k(
    qpp_score: float,
    tau1: float,
    tau2: float,
) -> int:
    """
    Return retrieval depth k(q) given predicted effectiveness ŷ(q).

    Monotonic relationship (Eq. 48):
        ŷ(q₁) < ŷ(q₂) ⟹ k(q₁) ≥ k(q₂)
    """
    if qpp_score >= tau2:
        return K_HIGH
    if qpp_score >= tau1:
        return K_MEDIUM
    return K_LOW


def compute_thresholds(qpp_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute τ₁ (33rd pct) and τ₂ (66th pct) from the predicted scores.
    (Eq. 45)
    """
    tau1 = float(np.percentile(qpp_scores, 33))
    tau2 = float(np.percentile(qpp_scores, 66))
    logger.info("Adaptive thresholds: τ₁=%.4f  τ₂=%.4f", tau1, tau2)
    return tau1, tau2


# ══════════════════════════════════════════════════════════════════════════════
# Learning-free baseline: query-length heuristic
# ══════════════════════════════════════════════════════════════════════════════

def length_based_k(query: str) -> int:
    """
    Depth heuristic used in Section 6 comparison:
      len < 5  → k=20
      5 ≤ len < 10 → k=30
      len ≥ 10 → k=50
    """
    n = len(query.split())
    if n < 5:
        return K_HIGH
    if n < 10:
        return K_MEDIUM
    return K_LOW


# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    query:       str
    qpp_score:   float
    k_used:      int
    passages:    List[str]
    answer:      str
    rouge_l:     float = 0.0
    bert_f1:     float = 0.0
    token_f1:    float = 0.0
    mrr:         float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveRAGPipeline:
    """
    End-to-end QPP-guided Adaptive RAG pipeline.

    Steps for each query:
      1. Initial retrieval (k=20 passages + embeddings)
      2. QPP feature extraction  →  predict ŷ(q)
      3. Determine adaptive k based on ŷ(q) and (τ₁, τ₂)
      4. If k > 20, re-retrieve with expanded depth
      5. Generate answer conditioned on top-5 passages
      6. (Optional) Evaluate against reference answer

    Usage:
        pipeline = AdaptiveRAGPipeline(retriever, generator, qpp_model, scaler)
        results  = pipeline.run(queries, references=references)
    """

    def __init__(
        self,
        retriever:      BaseRetriever,
        generator:      BaseGenerator,
        qpp_model,                          # fitted sklearn regressor
        scaler,                             # fitted StandardScaler
        df_lookup:      dict,
        corpus_lm:      dict,
        corpus_size:    int  = 8_841_823,
        query_emb_fn    = None,             # callable(query) → np.ndarray(768,)
        use_length_heuristic: bool = False, # toggle for ablation comparison
    ):
        self.retriever   = retriever
        self.generator   = generator
        self.qpp_model   = qpp_model
        self.scaler      = scaler
        self.df_lookup   = df_lookup
        self.corpus_lm   = corpus_lm
        self.corpus_size = corpus_size
        self.query_emb_fn = query_emb_fn
        self.use_length_heuristic = use_length_heuristic

    def _get_query_emb(self, query: str) -> np.ndarray:
        if self.query_emb_fn is not None:
            return self.query_emb_fn(query)
        # Fallback: zero vector (BM25-only setting)
        return np.zeros(768, dtype=np.float32)

    def _predict_qpp(
        self,
        query: str,
        passages: List[dict],
    ) -> float:
        """Extract features and return scalar QPP score."""
        texts   = [p["text"]      for p in passages]
        embs    = np.stack([p["embedding"] for p in passages])  # (k, 768)
        q_emb   = self._get_query_emb(query)                    # (768,)

        fv = extract_features(
            query, texts, embs, q_emb,
            self.df_lookup, self.corpus_lm, self.corpus_size
        ).reshape(1, -1)                                         # (1, 12)

        fv_norm  = self.scaler.transform(fv)
        score    = float(self.qpp_model.predict(fv_norm)[0])
        return score

    def run_query(
        self,
        query: str,
        reference: Optional[str] = None,
        tau1: float = 0.0,
        tau2: float = 1.0,
    ) -> QueryResult:
        """Process a single query through the adaptive pipeline."""

        # ── Step 1: initial retrieval (k=20) ──────────────────────────────
        initial_passages = self.retriever.retrieve(query, top_k=20)

        # ── Step 2: QPP score ─────────────────────────────────────────────
        if self.use_length_heuristic:
            qpp_score = 0.0   # unused; k determined by query length
            k = length_based_k(query)
        else:
            qpp_score = self._predict_qpp(query, initial_passages)
            k = adaptive_k(qpp_score, tau1, tau2)

        # ── Step 3: adaptive retrieval if k > 20 ─────────────────────────
        if k > 20:
            passages = self.retriever.retrieve(query, top_k=k)
        else:
            passages = initial_passages

        passage_texts = [p["text"] for p in passages]

        # ── Step 4: generation (top-5 passages as context) ───────────────
        answer = self.generator.generate(query, passage_texts[:5])

        result = QueryResult(
            query=query,
            qpp_score=qpp_score,
            k_used=k,
            passages=passage_texts,
            answer=answer,
        )

        # ── Step 5: evaluation (if reference available) ───────────────────
        if reference:
            from src.evaluate import rouge_l, token_f1, bert_score_f1
            result.rouge_l   = rouge_l(answer, reference)
            result.token_f1  = token_f1(answer, reference)
            bsf1             = bert_score_f1([answer], [reference])
            result.bert_f1   = bsf1[0]

        return result

    def run(
        self,
        queries:    List[str],
        references: Optional[List[str]] = None,
        verbose:    bool = True,
    ) -> List[QueryResult]:
        """
        Process a list of queries.

        If references are provided:
          - Compute generation metrics per query
          - Run stratified analysis (High/Medium/Low-QPP bins)
        """
        # Pre-compute thresholds from a first-pass QPP prediction
        if not self.use_length_heuristic:
            logger.info("Computing QPP thresholds from %d queries …", len(queries))
            prelim_scores = []
            prelim_passages_all = []
            for q in queries:
                psg = self.retriever.retrieve(q, top_k=20)
                prelim_passages_all.append(psg)
                prelim_scores.append(self._predict_qpp(q, psg))
            qpp_arr = np.array(prelim_scores)
            tau1, tau2 = compute_thresholds(qpp_arr)
        else:
            tau1, tau2 = 0.0, 1.0  # unused in heuristic mode

        results = []
        for i, query in enumerate(queries):
            ref = references[i] if references else None
            res = self.run_query(query, ref, tau1=tau1, tau2=tau2)
            results.append(res)
            if verbose and (i + 1) % 50 == 0:
                logger.info("Processed %d / %d queries", i + 1, len(queries))

        # ── Summary stats ─────────────────────────────────────────────────
        if references:
            _log_summary(results)

        return results


def _log_summary(results: List[QueryResult]) -> None:
    rouge  = np.mean([r.rouge_l  for r in results])
    bert   = np.mean([r.bert_f1  for r in results])
    f1     = np.mean([r.token_f1 for r in results])
    logger.info(
        "─── Summary ───  ROUGE-L=%.4f | BERTScore-F1=%.4f | Token-F1=%.4f",
        rouge, bert, f1
    )
    k_dist = {}
    for r in results:
        k_dist[r.k_used] = k_dist.get(r.k_used, 0) + 1
    logger.info("Retrieval depth distribution: %s", k_dist)
