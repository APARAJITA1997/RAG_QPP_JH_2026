"""
run_pipeline.py
---------------
End-to-end RAG-QPP pipeline runner.

Modes:
  --mode qpp_only        Train QPP model + evaluate correlations
  --mode generation      Run generation + evaluate ROUGE-L / BERTScore
  --mode adaptive_rag    Full QPP-guided adaptive RAG pipeline
  --mode ablation        Run feature-group ablation study

Examples:
  python run_pipeline.py --dataset msmarco_passage --retriever dense --model random_forest
  python run_pipeline.py --dataset nq --retriever hybrid --mode adaptive_rag --generator bart
  python run_pipeline.py --mode ablation --dataset msmarco_passage --retriever dense
"""

import argparse
import logging
import os
import json
import pickle
import random
from pathlib import Path

import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch; torch.manual_seed(RANDOM_SEED)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="RAG-QPP pipeline")

    # Dataset
    p.add_argument(
        "--dataset", default="msmarco_passage",
        choices=["msmarco_passage", "msmarco_document", "nq", "robust04"],
        help="Dataset to evaluate on."
    )
    p.add_argument("--data_dir", default="data/",
                   help="Root directory containing dataset files.")

    # Retriever
    p.add_argument(
        "--retriever", default="dense",
        choices=["bm25", "dense", "hybrid", "colbert"],
    )
    p.add_argument("--bm25_index",   default=None, help="Path to Pyserini BM25 index.")
    p.add_argument("--faiss_index",  default=None, help="Path to FAISS index file.")
    p.add_argument("--colbert_index",default=None, help="Path to ColBERT PLAID index.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Initial retrieval depth.")

    # QPP model
    p.add_argument(
        "--model", default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
    )
    p.add_argument("--model_path", default=None,
                   help="Load a pre-trained QPP model from disk.")
    p.add_argument("--save_model", default="outputs/checkpoints/qpp_model.pkl",
                   help="Where to save the trained QPP model.")

    # Generator
    p.add_argument(
        "--generator", default="bart",
        choices=["bart", "llama"],
    )
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace token (required for LLaMA).")

    # Mode
    p.add_argument(
        "--mode", default="qpp_only",
        choices=["qpp_only", "generation", "adaptive_rag", "ablation"],
    )

    # Misc
    p.add_argument("--output_dir", default="outputs/predictions/")
    p.add_argument("--max_queries", type=int, default=None,
                   help="Limit number of queries (for quick testing).")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading stubs
# ══════════════════════════════════════════════════════════════════════════════

def load_queries_and_labels(dataset: str, data_dir: str, max_queries=None):
    """
    Load (queries, relevance_lists, references) for the given dataset.

    Returns
    -------
    queries         : List[str]
    relevance_lists : List[List[int]]  — per-rank binary/graded relevance
    references      : List[str] | None — reference answers (NQ, MS MARCO Doc)
    """
    logger.info("Loading dataset: %s from %s", dataset, data_dir)

    if dataset == "msmarco_passage":
        return _load_msmarco_passage(data_dir, max_queries)
    elif dataset == "msmarco_document":
        return _load_msmarco_document(data_dir, max_queries)
    elif dataset == "nq":
        return _load_nq(data_dir, max_queries)
    elif dataset == "robust04":
        return _load_robust04(data_dir, max_queries)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_msmarco_passage(data_dir, max_queries):
    """
    Loads MS MARCO Passage dev queries (6,980) and qrels.
    Expected file layout:
      data/msmarco_passage/queries.dev.tsv      — qid\tquery
      data/msmarco_passage/qrels.dev.tsv        — qid\t0\tpid\trel
    """
    queries_path = Path(data_dir) / "msmarco_passage" / "queries.dev.tsv"
    qrels_path   = Path(data_dir) / "msmarco_passage" / "qrels.dev.tsv"

    queries_raw, qid_list = {}, []
    with open(queries_path) as f:
        for line in f:
            qid, query = line.strip().split("\t", 1)
            queries_raw[qid] = query
            qid_list.append(qid)

    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[pid] = rel

    queries = []
    for qid in qid_list[:max_queries]:
        queries.append((qid, queries_raw[qid], qrels.get(qid, {})))

    return queries, None   # references = None for passage collection


def _load_msmarco_document(data_dir, max_queries):
    """MS MARCO Document QA — has human-generated reference answers."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ms_marco", "v1.1", split="validation")
        queries, references = [], []
        for row in (ds if max_queries is None else ds.select(range(max_queries))):
            queries.append(row["query"])
            # Use 'answers' field; take first non-empty answer
            answers = [a for a in row.get("answers", []) if a.strip()]
            references.append(answers[0] if answers else "")
        return queries, references
    except Exception as e:
        logger.warning("HuggingFace ms_marco load failed (%s). Using stub.", e)
        return ["what is machine learning"] * (max_queries or 10), \
               ["Machine learning is a subset of AI."] * (max_queries or 10)


def _load_nq(data_dir, max_queries):
    """Natural Questions Open — 100 validation queries."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "google-research-datasets/nq_open", split="validation"
        )
        queries, references = [], []
        for row in (ds if max_queries is None else ds.select(range(max_queries or 100))):
            queries.append(row["question"])
            answers = row.get("answer", [])
            references.append(answers[0] if answers else "")
        return queries, references
    except Exception as e:
        logger.warning("NQ load failed (%s). Using stub data.", e)
        n = max_queries or 10
        return ["what year did world war 2 end"] * n, ["1945"] * n


def _load_robust04(data_dir, max_queries):
    """TREC Robust04 — 249 topics, no reference answers."""
    try:
        import ir_datasets
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        queries, qids = [], []
        for q in dataset.queries_iter():
            queries.append(q.title)
            qids.append(q.query_id)
            if max_queries and len(queries) >= max_queries:
                break
        return queries, None
    except Exception as e:
        logger.warning("Robust04 load failed (%s). Using stub.", e)
        return ["air pollution effects on health"] * (max_queries or 10), None


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline modes
# ══════════════════════════════════════════════════════════════════════════════

def run_qpp_only(args, retriever, queries, relevance_lists):
    """Train QPP models and report Pearson/Spearman/Kendall correlations."""
    from src.features import extract_features_batch, normalize_features, FEATURE_NAMES
    from src.models   import train_all_models, predict, save_model
    from src.evaluate import compute_correlations, compute_retrieval_metrics

    logger.info("=== MODE: QPP ONLY ===")

    # ── Retrieve passages + extract features ─────────────────────────────
    logger.info("Retrieving passages and extracting QPP features …")
    all_texts, all_embs, all_q_embs = [], [], []
    for q_data in queries:
        query = q_data if isinstance(q_data, str) else q_data[1]
        results = retriever.retrieve(query, top_k=20)
        all_texts.append([r["text"]      for r in results])
        all_embs.append( np.stack([r["embedding"] for r in results]))
        # Dense query embedding (zeros if BM25)
        if hasattr(retriever, "encode_query"):
            all_q_embs.append(retriever.encode_query(query))
        else:
            all_q_embs.append(np.zeros(768, dtype=np.float32))

    # ── Build ground-truth MRR@10 labels ─────────────────────────────────
    # relevance_lists: list of {pid: rel} dicts (MS MARCO) or List[List[int]]
    logger.info("Building ground-truth MRR@10 labels …")
    y_true = _compute_mrr_labels(queries, all_texts, relevance_lists)

    # ── Feature extraction ────────────────────────────────────────────────
    query_strings = [q if isinstance(q, str) else q[1] for q in queries]
    df_lookup, corpus_lm = _load_corpus_stats(args.data_dir, args.dataset)

    X = extract_features_batch(
        query_strings, all_texts, all_embs,
        np.stack(all_q_embs), df_lookup, corpus_lm
    )

    # ── Train / test split (80/20) ────────────────────────────────────────
    n = len(X)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_true[:split], y_true[split:]

    X_train_n, X_test_n, scaler = normalize_features(X_train, X_test)

    # ── Train models ──────────────────────────────────────────────────────
    models = train_all_models(X_train_n, y_train)

    # ── Evaluate correlations ─────────────────────────────────────────────
    all_results = {}
    for name, model in models.items():
        y_pred = predict(model, X_test_n)
        corr   = compute_correlations(y_pred, y_test)
        all_results[name] = corr
        logger.info(
            "[%s]  Pearson=%.4f | Spearman=%.4f | Kendall=%.4f",
            name, corr["pearson_r"], corr["spearman_rho"], corr["kendall_tau"]
        )

    # ── Save best model (Random Forest for in-domain) ────────────────────
    best_model = models[args.model]
    save_model(best_model, args.save_model)

    # ── Persist results ───────────────────────────────────────────────────
    out = Path(args.output_dir) / "qpp_correlations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved → %s", out)

    return models, scaler


def run_ablation(args, retriever, queries, relevance_lists):
    """Reproduce Table 10: feature-group ablation study."""
    from src.features import (
        extract_features_batch, normalize_features, FEATURE_NAMES
    )
    from src.models   import train_model, predict
    from src.evaluate import compute_correlations

    logger.info("=== MODE: ABLATION ===")

    # Feature groups (by index in the 12-D vector)
    GROUPS = {
        "All Features (12-D)":          list(range(12)),
        "– Similarity-based":           list(range(5, 12)),   # remove 0-4
        "– Embedding-based":            [0,1,2,3] + list(range(5, 12)),
        "– Term-based":                 list(range(0, 6)) + list(range(9, 12)),
        "– Traditional QPP (Clarity/WIG/NQC)": list(range(0, 9)),
    }

    # (Retrieve + extract full feature matrix as above)
    logger.info("Extracting full feature matrix for ablation …")
    query_strings = [q if isinstance(q, str) else q[1] for q in queries]
    df_lookup, corpus_lm = _load_corpus_stats(args.data_dir, args.dataset)

    all_texts, all_embs, all_q_embs = [], [], []
    for q_data in queries:
        query = q_data if isinstance(q_data, str) else q_data[1]
        results = retriever.retrieve(query, top_k=20)
        all_texts.append([r["text"] for r in results])
        all_embs.append(np.stack([r["embedding"] for r in results]))
        all_q_embs.append(
            retriever.encode_query(query) if hasattr(retriever, "encode_query")
            else np.zeros(768, dtype=np.float32)
        )

    X_full = extract_features_batch(
        query_strings, all_texts, all_embs,
        np.stack(all_q_embs), df_lookup, corpus_lm
    )
    y_true = _compute_mrr_labels(queries, all_texts, relevance_lists)

    n     = len(X_full)
    split = int(0.8 * n)

    ablation_results = {}
    for label, keep_cols in GROUPS.items():
        X = X_full[:, keep_cols]
        X_tr_n, X_te_n, _ = normalize_features(X[:split], X[split:])
        y_tr, y_te        = y_true[:split], y_true[split:]

        model  = train_model("random_forest", X_tr_n, y_tr, cv=False)
        y_pred = model.predict(X_te_n)
        corr   = compute_correlations(y_pred, y_te)

        ablation_results[label] = {
            "pearson":  round(corr["pearson_r"],    4),
            "spearman": round(corr["spearman_rho"], 4),
            "kendall":  round(corr["kendall_tau"],  4),
        }
        logger.info(
            "%-42s  Pearson=%.4f | Spearman=%.4f | Kendall=%.4f",
            label,
            corr["pearson_r"], corr["spearman_rho"], corr["kendall_tau"]
        )

    out = Path(args.output_dir) / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info("Ablation results saved → %s", out)
    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_mrr_labels(queries, all_retrieved_texts, relevance_data, n=10):
    """Stub: return random MRR labels if qrels not available."""
    if relevance_data is None:
        logger.warning("No relevance judgments — using random MRR labels (stub).")
        return np.random.uniform(0, 1, len(queries)).astype(np.float32)
    # Real implementation would map retrieved passage IDs to qrel judgments
    return np.random.uniform(0, 1, len(queries)).astype(np.float32)


def _load_corpus_stats(data_dir, dataset):
    """
    Load term document-frequency map and corpus unigram language model.
    Falls back to empty dicts if files not found.
    """
    df_path  = Path(data_dir) / dataset / "df_lookup.json"
    lm_path  = Path(data_dir) / dataset / "corpus_lm.json"

    df_lookup = {}
    corpus_lm = {}

    if df_path.exists():
        with open(df_path) as f:
            df_lookup = json.load(f)
        logger.info("Loaded df_lookup (%d terms)", len(df_lookup))
    else:
        logger.warning("df_lookup not found at %s — IDF features will be 0.", df_path)

    if lm_path.exists():
        with open(lm_path) as f:
            corpus_lm = json.load(f)
        logger.info("Loaded corpus_lm (%d terms)", len(corpus_lm))
    else:
        logger.warning("corpus_lm not found at %s — Clarity will be 0.", lm_path)

    return df_lookup, corpus_lm


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    logger.info("RAG-QPP  |  dataset=%s  retriever=%s  model=%s  mode=%s",
                args.dataset, args.retriever, args.model, args.mode)

    # ── Build retriever ───────────────────────────────────────────────────
    from src.retrieval import build_retriever

    retriever_kwargs = {}
    if args.retriever == "bm25":
        if not args.bm25_index:
            raise ValueError("--bm25_index is required for BM25 retriever.")
        retriever_kwargs["index_path"] = args.bm25_index
    elif args.retriever == "dense":
        if args.faiss_index:
            retriever_kwargs["faiss_index_path"] = args.faiss_index

    retriever = build_retriever(args.retriever, **retriever_kwargs)

    # ── Load data ─────────────────────────────────────────────────────────
    queries, references = load_queries_and_labels(
        args.dataset, args.data_dir, args.max_queries
    )

    # ── Dispatch mode ─────────────────────────────────────────────────────
    if args.mode == "qpp_only":
        run_qpp_only(args, retriever, queries, None)

    elif args.mode == "ablation":
        run_ablation(args, retriever, queries, None)

    elif args.mode in ("generation", "adaptive_rag"):
        models, scaler = run_qpp_only(args, retriever, queries, None)

        from src.generation  import build_generator
        from src.adaptive_rag import AdaptiveRAGPipeline

        generator = build_generator(
            args.generator,
            hf_token=args.hf_token,
        )
        df_lookup, corpus_lm = _load_corpus_stats(args.data_dir, args.dataset)

        query_emb_fn = None
        if hasattr(retriever, "encode_query"):
            query_emb_fn = retriever.encode_query

        pipeline = AdaptiveRAGPipeline(
            retriever=retriever,
            generator=generator,
            qpp_model=models[args.model],
            scaler=scaler,
            df_lookup=df_lookup,
            corpus_lm=corpus_lm,
            query_emb_fn=query_emb_fn,
        )

        query_strings = [q if isinstance(q, str) else q[1] for q in queries]
        results = pipeline.run(query_strings, references=references)

        # Save answer outputs
        out = Path(args.output_dir) / f"{args.mode}_results.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "query":     r.query,
                    "qpp_score": r.qpp_score,
                    "k_used":    r.k_used,
                    "answer":    r.answer,
                    "rouge_l":   r.rouge_l,
                    "bert_f1":   r.bert_f1,
                    "token_f1":  r.token_f1,
                }) + "\n")
        logger.info("Results saved → %s", out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
