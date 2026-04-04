"""
src/evaluate.py
---------------
Evaluation metrics for RAG-QPP:

  Retrieval-level:
    - MRR@n  (Eq. 32-33) — primary QPP target
    - nDCG@k (Eq. 34-35)
    - AP      (Eq. 36)

  QPP correlation:
    - Pearson r    (linear association)
    - Spearman ρ   (rank ordering)
    - Kendall τ    (pairwise concordance)

  Generation-level:
    - ROUGE-L       (Eq. 37) — longest common subsequence overlap
    - BERTScore-F1  (Eq. 38) — contextual embedding similarity
    - Token-level F1 (Eq. 39)

Reference: Sinha & Chakma (2026), Section 3.14
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval Metrics
# ══════════════════════════════════════════════════════════════════════════════

def reciprocal_rank(relevance_list: List[int], n: int = 10) -> float:
    """
    Reciprocal rank of the first relevant item within the top-n results.
    RR@n = 1/rank_q  if a relevant passage appears in top-n, else 0.  (Eq. 32)
    """
    for i, rel in enumerate(relevance_list[:n]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    relevance_lists: List[List[int]], n: int = 10
) -> float:
    """
    MRR@n over a list of queries.  (Eq. 33)
    """
    rrs = [reciprocal_rank(rl, n) for rl in relevance_lists]
    return float(np.mean(rrs))


def ndcg_at_k(relevance_list: List[int], k: int = 10) -> float:
    """
    nDCG@k using graded relevance.  (Eqs. 34-35)
    """
    def dcg(rels: List[int], cutoff: int) -> float:
        return sum(
            (2 ** r - 1) / math.log2(i + 2)
            for i, r in enumerate(rels[:cutoff])
        )

    actual_dcg  = dcg(relevance_list, k)
    ideal_dcg   = dcg(sorted(relevance_list, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(relevance_list: List[int]) -> float:
    """
    Average Precision (AP) over all retrieved results.  (Eq. 36)
    """
    num_relevant = sum(1 for r in relevance_list if r > 0)
    if num_relevant == 0:
        return 0.0

    running_hits, ap = 0, 0.0
    for i, rel in enumerate(relevance_list):
        if rel > 0:
            running_hits += 1
            ap += running_hits / (i + 1)
    return ap / num_relevant


def compute_retrieval_metrics(
    relevance_lists: List[List[int]],
    mrr_cutoffs: Tuple[int, ...] = (5, 10, 20, 50),
    ndcg_cutoffs: Tuple[int, ...] = (10, 100),
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for a set of queries.

    Returns a flat dict, e.g.:
      {"MRR@5": 0.91, "MRR@10": 0.95, ..., "nDCG@10": 0.47, "AP": 0.42}
    """
    metrics: Dict[str, float] = {}

    for n in mrr_cutoffs:
        metrics[f"MRR@{n}"] = mean_reciprocal_rank(relevance_lists, n=n)

    for k in ndcg_cutoffs:
        scores = [ndcg_at_k(rl, k) for rl in relevance_lists]
        metrics[f"nDCG@{k}"] = float(np.mean(scores))

    ap_scores = [average_precision(rl) for rl in relevance_lists]
    metrics["AP"] = float(np.mean(ap_scores))

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# QPP Correlation  (Eqs. 62-66)
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlations(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Pearson r, Spearman ρ, and Kendall τ between predicted and
    ground-truth retrieval effectiveness scores.

    All three are reported together as recommended in the paper (Section 5.2)
    because each captures a different aspect of agreement:
      - Pearson r  : linear association (operational weight for RAG)
      - Spearman ρ : rank ordering consistency
      - Kendall τ  : pairwise concordance (most direct ranking fidelity)

    Returns
    -------
    Dict with keys: "pearson_r", "pearson_p",
                    "spearman_rho", "spearman_p",
                    "kendall_tau", "kendall_p"
    """
    pr, pp   = pearsonr(y_pred, y_true)
    sr, sp   = spearmanr(y_pred, y_true)
    kt, kp   = kendalltau(y_pred, y_true)

    results = {
        "pearson_r":    float(pr),
        "pearson_p":    float(pp),
        "spearman_rho": float(sr),
        "spearman_p":   float(sp),
        "kendall_tau":  float(kt),
        "kendall_p":    float(kp),
    }

    logger.info(
        "Correlations → Pearson r=%.4f (p=%.4f) | "
        "Spearman ρ=%.4f (p=%.4f) | Kendall τ=%.4f (p=%.4f)",
        pr, pp, sr, sp, kt, kp
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Generation Metrics
# ══════════════════════════════════════════════════════════════════════════════

def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L score using longest common subsequence (LCS).  (Eq. 37)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        score  = scorer.score(reference, hypothesis)
        return float(score["rougeL"].fmeasure)
    except ImportError:
        # Manual LCS fallback
        h_tokens = hypothesis.lower().split()
        r_tokens = reference.lower().split()
        lcs = _lcs_length(h_tokens, r_tokens)
        if len(r_tokens) == 0:
            return 0.0
        return lcs / len(r_tokens)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Dynamic-programming LCS length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def token_f1(hypothesis: str, reference: str) -> float:
    """
    Token-level F1: overlap between generated and reference answer tokens.
    (Eq. 39)
    """
    h_tokens = set(hypothesis.lower().split())
    r_tokens = set(reference.lower().split())
    if not h_tokens or not r_tokens:
        return 0.0
    common    = h_tokens & r_tokens
    precision = len(common) / len(h_tokens)
    recall    = len(common) / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bert_score_f1(
    hypotheses: List[str],
    references: List[str],
    model_type: str = "roberta-large",
    device: Optional[str] = None,
) -> List[float]:
    """
    BERTScore-F1 using contextual embeddings.  (Eq. 38)

    Args:
        hypotheses : list of generated answers.
        references : list of reference answers.
        model_type : backbone model for BERTScore (default roberta-large).
        device     : "cuda" or "cpu".

    Returns
    -------
    List of per-query BERTScore-F1 values.
    """
    try:
        from bert_score import score as bs_score
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _, _, F = bs_score(
            hypotheses, references,
            model_type=model_type,
            device=dev,
            verbose=False,
        )
        return F.tolist()
    except ImportError:
        logger.warning(
            "bert_score not installed. Returning 0.0 for all queries. "
            "Install with:  pip install bert-score"
        )
        return [0.0] * len(hypotheses)


def compute_generation_metrics(
    hypotheses: List[str],
    references: List[str],
    compute_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Compute ROUGE-L, BERTScore-F1, and token-level F1 for a list of
    generated / reference pairs.

    Returns
    -------
    Dict: {"rouge_l": float, "bert_f1": float, "token_f1": float}
    """
    rl_scores  = [rouge_l(h, r)    for h, r in zip(hypotheses, references)]
    f1_scores  = [token_f1(h, r)   for h, r in zip(hypotheses, references)]

    result = {
        "rouge_l":   float(np.mean(rl_scores)),
        "token_f1":  float(np.mean(f1_scores)),
    }

    if compute_bertscore:
        bs_scores = bert_score_f1(hypotheses, references)
        result["bert_f1"] = float(np.mean(bs_scores))
    else:
        result["bert_f1"] = 0.0

    logger.info(
        "Generation metrics → ROUGE-L=%.4f | BERTScore-F1=%.4f | Token-F1=%.4f",
        result["rouge_l"], result["bert_f1"], result["token_f1"]
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# QPP-Bin stratified analysis  (Table 7 reproduction)
# ══════════════════════════════════════════════════════════════════════════════

def stratified_analysis(
    qpp_scores:   np.ndarray,
    mrr_scores:   np.ndarray,
    rouge_scores: np.ndarray,
    bert_scores:  np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Partition queries into High / Medium / Low-QPP bins using tertile
    thresholds (τ₁ = 33rd pct, τ₂ = 66th pct) and compute per-bin averages.

    Returns
    -------
    Nested dict: {"High": {"mrr": .., "rouge_l": .., "bert_f1": ..}, ...}
    """
    tau1 = float(np.percentile(qpp_scores, 33))
    tau2 = float(np.percentile(qpp_scores, 66))

    bins = {
        "High":   qpp_scores >= tau2,
        "Medium": (qpp_scores >= tau1) & (qpp_scores < tau2),
        "Low":    qpp_scores < tau1,
    }

    results = {}
    for label, mask in bins.items():
        results[label] = {
            "n":       int(mask.sum()),
            "mrr":     float(mrr_scores[mask].mean())   if mask.any() else 0.0,
            "rouge_l": float(rouge_scores[mask].mean()) if mask.any() else 0.0,
            "bert_f1": float(bert_scores[mask].mean())  if mask.any() else 0.0,
        }
        logger.info(
            "%-6s QPP bin (n=%3d): MRR=%.3f | ROUGE-L=%.3f | BERT-F1=%.3f",
            label, results[label]["n"],
            results[label]["mrr"],
            results[label]["rouge_l"],
            results[label]["bert_f1"],
        )

    return results
