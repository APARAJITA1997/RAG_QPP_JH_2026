"""
src/retrieval.py
----------------
Four retrieval paradigms evaluated in RAG-QPP:

  1. BM25          — sparse lexical matching via Pyserini
  2. Dense         — bi-encoder Sentence-BERT + FAISS flat index
  3. Hybrid        — min-max fused BM25 + Dense scores (α = 0.5)
  4. ColBERT       — late-interaction neural retrieval via ColBERT-AI

Each retriever returns the same output format:
    List[dict] with keys: {"id", "text", "score", "embedding"}

Reference: Sinha & Chakma (2026), Sections 3.11, 4
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared result type
# ══════════════════════════════════════════════════════════════════════════════

PassageResult = Dict  # keys: id, text, score, embedding (np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class BaseRetriever(ABC):
    """All retrievers implement this interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> List[PassageResult]:
        """
        Retrieve top_k passages for the given query.

        Returns a list of dicts (sorted descending by score):
            {"id": str, "text": str, "score": float, "embedding": np.ndarray}
        """

    def batch_retrieve(
        self, queries: List[str], top_k: int = 20
    ) -> List[List[PassageResult]]:
        return [self.retrieve(q, top_k) for q in queries]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BM25 — Sparse lexical retriever (Pyserini)
# ══════════════════════════════════════════════════════════════════════════════

class BM25Retriever(BaseRetriever):
    """
    BM25 sparse retrieval via Pyserini.
    k1 = 0.9, b = 0.4  (paper defaults, Section 4).

    The embedding field is set to zeros because BM25 has no dense
    representation; the QPP feature extractor falls back to tf-idf
    signals for such passages.

    Usage:
        retriever = BM25Retriever(index_path="/path/to/lucene_index")
        results   = retriever.retrieve("when was the burj khalifa built")
    """

    def __init__(self, index_path: str, k1: float = 0.9, b: float = 0.4):
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError(
                "Pyserini is required for BM25 retrieval. "
                "Install it with:  pip install pyserini"
            )
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)
        logger.info("BM25 index loaded from %s (k1=%.2f, b=%.2f)", index_path, k1, b)

    def retrieve(self, query: str, top_k: int = 20) -> List[PassageResult]:
        hits = self.searcher.search(query, k=top_k)
        results = []
        for hit in hits:
            doc  = self.searcher.doc(hit.docid)
            text = doc.raw() if doc else ""
            results.append({
                "id":        hit.docid,
                "text":      text,
                "score":     float(hit.score),
                "embedding": np.zeros(768, dtype=np.float32),  # BM25 has no embedding
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Dense bi-encoder  (Sentence-BERT + FAISS)
# ══════════════════════════════════════════════════════════════════════════════

class DenseRetriever(BaseRetriever):
    """
    Dense bi-encoder retrieval using Sentence-BERT.
    Default checkpoint: 'msmarco-distilbert-base-v4'  (Section 4).
    Embedding dimension: 768.
    Index: FAISS Flat (exact L2 / cosine via normalized vectors).

    Build the index once with `build_index`, then call `retrieve`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            raise ImportError(
                "sentence-transformers and faiss-cpu are required. "
                "Install with:  pip install sentence-transformers faiss-cpu"
            )
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.faiss      = faiss
        self.index      = None
        self.passage_ids: List[str]    = []
        self.passage_texts: List[str]  = []
        logger.info("Dense encoder loaded: %s  (device=%s)", model_name, self.device)

    def _encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    def build_index(self, passage_ids: List[str], passage_texts: List[str]) -> None:
        """
        Encode all passages and build a FAISS Flat inner-product index.
        After L2 normalization, inner product == cosine similarity.
        """
        logger.info("Encoding %d passages …", len(passage_texts))
        embs = self._encode(passage_texts, normalize=True)         # (N, 768)

        d     = embs.shape[1]
        index = self.faiss.IndexFlatIP(d)                          # inner product
        index.add(embs)

        self.index         = index
        self.passage_ids   = passage_ids
        self.passage_texts = passage_texts
        self._all_embs     = embs                                   # kept for QPP features
        logger.info("FAISS index built: %d passages, dim=%d", len(passage_ids), d)

    def save_index(self, path: str) -> None:
        self.faiss.write_index(self.index, path)
        logger.info("FAISS index saved to %s", path)

    def load_index(self, path: str,
                   passage_ids: List[str],
                   passage_texts: List[str],
                   all_embs: np.ndarray) -> None:
        self.index         = self.faiss.read_index(path)
        self.passage_ids   = passage_ids
        self.passage_texts = passage_texts
        self._all_embs     = all_embs
        logger.info("FAISS index loaded from %s", path)

    def retrieve(self, query: str, top_k: int = 20) -> List[PassageResult]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb   = self._encode([query], normalize=True)[0]         # (768,)
        scores, indices = self.index.search(
            q_emb.reshape(1, -1), top_k
        )                                                           # (1, top_k)
        scores  = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:                                             # FAISS OOB
                continue
            results.append({
                "id":        self.passage_ids[idx],
                "text":      self.passage_texts[idx],
                "score":     float(score),
                "embedding": self._all_embs[idx],
            })
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Return (768,) normalized query embedding."""
        return self._encode([query], normalize=True)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hybrid retriever  (BM25 + Dense, equal weights)
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever(BaseRetriever):
    """
    Linear fusion of BM25 and Dense scores (Eq. 43):
        s_hybrid = 0.5 * ŝ_BM25 + 0.5 * ŝ_dense
    where ŝ denotes min-max normalised scores per query.

    alpha = 0.5 provides equal weight to lexical and semantic evidence.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        candidate_k: int = 100,
    ):
        self.bm25   = bm25_retriever
        self.dense  = dense_retriever
        self.alpha  = alpha
        self.candidate_k = candidate_k

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def retrieve(self, query: str, top_k: int = 20) -> List[PassageResult]:
        # Fetch candidate results from both retrievers
        bm25_results  = self.bm25.retrieve(query,  top_k=self.candidate_k)
        dense_results = self.dense.retrieve(query, top_k=self.candidate_k)

        # Build {doc_id: score} maps
        bm25_scores  = {r["id"]: r["score"] for r in bm25_results}
        dense_scores = {r["id"]: r["score"] for r in dense_results}
        dense_embs   = {r["id"]: r["embedding"] for r in dense_results}
        dense_texts  = {r["id"]: r["text"]      for r in dense_results}

        all_ids = list(set(bm25_scores) | set(dense_scores))

        # Min-max normalize each score series
        bm25_arr  = np.array([bm25_scores.get(i,  0.0) for i in all_ids])
        dense_arr = np.array([dense_scores.get(i, 0.0) for i in all_ids])

        bm25_norm  = self._minmax(bm25_arr)
        dense_norm = self._minmax(dense_arr)

        fused = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

        # Sort and take top_k
        order   = np.argsort(-fused)[:top_k]
        results = []
        for idx in order:
            pid = all_ids[idx]
            results.append({
                "id":        pid,
                "text":      dense_texts.get(pid, bm25_scores.get(pid, "")),
                "score":     float(fused[idx]),
                "embedding": dense_embs.get(pid, np.zeros(768, dtype=np.float32)),
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ColBERT  (late-interaction, token-level MaxSim)
# ══════════════════════════════════════════════════════════════════════════════

class ColBERTRetriever(BaseRetriever):
    """
    ColBERTv2 late-interaction retrieval via the colbert-ai package.
    Checkpoint: 'colbert-ir/colbertv2.0'  (Section 4).
    Token embedding dim: 128.
    Index type: PLAID (IVF-PQ approximate + exact late-interaction reranking).

    MaxSim operator (Eq. 44):
        ColBERT(q, p) = Σ_t max_j (q_t · p_j)

    Usage:
        retriever = ColBERTRetriever(index_path="/path/to/plaid_index")
        results   = retriever.retrieve("what is photosynthesis")
    """

    def __init__(
        self,
        index_path: str,
        checkpoint: str = "colbert-ir/colbertv2.0",
        n_bits: int = 2,
    ):
        try:
            from colbert import Searcher
            from colbert.infra import ColBERTConfig
        except ImportError:
            raise ImportError(
                "colbert-ai is required for ColBERT retrieval. "
                "Install with:  pip install colbert-ai"
            )
        config = ColBERTConfig(
            checkpoint=checkpoint,
            index_path=index_path,
            nbits=n_bits,
        )
        self.searcher   = Searcher(index=index_path, config=config)
        self._checkpoint = checkpoint
        logger.info("ColBERT searcher loaded (checkpoint=%s, index=%s)",
                    checkpoint, index_path)

    def retrieve(self, query: str, top_k: int = 20) -> List[PassageResult]:
        results_raw = self.searcher.search(query, k=top_k)
        passage_ids, ranks, scores = results_raw

        results = []
        for pid, rank, score in zip(passage_ids, ranks, scores):
            doc_text = self.searcher.collection[pid]
            results.append({
                "id":        str(pid),
                "text":      doc_text,
                "score":     float(score),
                # ColBERT does not expose single-vector passage embeddings
                # through the PLAID interface; use zeros as placeholder.
                "embedding": np.zeros(768, dtype=np.float32),
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_retriever(
    retriever_type: str,
    **kwargs,
) -> BaseRetriever:
    """
    Factory function to instantiate a retriever by name.

    Parameters
    ----------
    retriever_type : one of {"bm25", "dense", "hybrid", "colbert"}.
    **kwargs       : forwarded to the chosen retriever's __init__.

    Example
    -------
    retriever = build_retriever(
        "dense",
        model_name="sentence-transformers/msmarco-distilbert-base-v4",
    )
    """
    mapping = {
        "bm25":    BM25Retriever,
        "dense":   DenseRetriever,
        "hybrid":  HybridRetriever,
        "colbert": ColBERTRetriever,
    }
    key = retriever_type.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown retriever '{retriever_type}'. "
            f"Choose from: {list(mapping)}"
        )
    return mapping[key](**kwargs)
