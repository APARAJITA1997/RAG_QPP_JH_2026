# RAG-QPP: Adaptive Query Performance Prediction for Retrieval-Augmented Generation

> **Paper:** *Adaptive Query Performance Prediction for Retrieval-Augmented Generation: Bridging Retrieval Quality and Generation Relevance*
> **Authors:** Aparajita Sinha & Kunal Chakma — National Institute of Technology Agartala, India
> **Venue:** ACM Transactions on Information Systems (Under Review)

---

## 📌 Overview

**RAG-QPP** is a retrieval-centric Query Performance Prediction (QPP) framework designed for Retrieval-Augmented Generation (RAG) pipelines. It predicts query difficulty from a **12-dimensional post-retrieval feature set** — combining semantic similarity, lexical, and score-distribution signals — without requiring relevance judgments or generator-internal signals at inference time.

```
Query → Retriever → Top-k Passages → QPP Features → Regressor → Predicted MRR@10
                                                                        ↓
                                               Adaptive Retrieval Depth Adjustment
                                                                        ↓
                                                         Generator → Answer
```

### Key Highlights
- ✅ **Architecture-agnostic** — works with any black-box generator (BART, LLaMA, etc.)
- ✅ **Retrieval-centric** — uses only post-retrieval signals, no perplexity/token uncertainty needed
- ✅ **Multi-paradigm** — evaluated on BM25, Dense (Sentence-BERT), Hybrid, ColBERT
- ✅ **Cross-domain** — generalizes from MS MARCO → Natural Questions and Robust04
- ✅ **Adaptive** — QPP score guides dynamic retrieval depth (k=20/30/50)

---

## 📊 Results Summary

| Model | MS MARCO Passage (Pearson r) | NQ (Pearson r) |
|---|---|---|
| Clarity (Baseline) | -0.035 | -0.152 |
| WIG (Baseline) | 0.147 | 0.179 |
| NQC (Baseline) | 0.436 | 0.580 |
| **Random Forest (RAG-QPP)** | **0.659** | 0.533 |
| XGBoost (RAG-QPP) | 0.560 | 0.367 |
| LightGBM (RAG-QPP) | 0.242 | **0.540** |

**QPP-Guided vs Baseline RAG:**

| Metric | Baseline RAG | QPP-Guided RAG | Gain |
|---|---|---|---|
| ROUGE-L | 0.360 | 0.383 | +0.023 |
| BERTScore | 0.437 | 0.459 | +0.022 |
| F1 | 0.400 | 0.423 | +0.023 |

---

## 🗂️ Repository Structure

```
rag-qpp/
│
├── README.md
├── requirements.txt
├── setup.py
├── run_pipeline.py               # End-to-end pipeline runner
│
├── src/
│   ├── __init__.py
│   ├── features.py               # 12-D QPP feature extraction
│   ├── retrieval.py              # BM25, Dense, Hybrid, ColBERT retrievers
│   ├── models.py                 # RF, XGBoost, LightGBM QPP regressors
│   ├── generation.py             # BART-large & LLaMA-3-8B generation
│   ├── adaptive_rag.py           # QPP-guided adaptive retrieval
│   └── evaluate.py               # Pearson, Spearman, Kendall, ROUGE-L, BERTScore
│
├── notebooks/
│   ├── 01_retrieval_analysis.ipynb
│   ├── 02_qpp_prediction.ipynb
│   ├── 03_retrieval_generation_correlation.ipynb
│   ├── 04_feature_ablation.ipynb
│   └── 05_adaptive_rag.ipynb
│
├── configs/
│   └── config.yaml               # All hyperparameters in one place
│
├── data/
│   └── README.md                 # Instructions for downloading datasets
│
└── outputs/
    ├── predictions/
    ├── figures/
    └── checkpoints/
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-qpp.git
cd rag-qpp
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Install ColBERT for late-interaction retrieval
```bash
pip install colbert-ai
```

---

## 📦 Datasets

| Dataset | Domain | Size | Use |
|---|---|---|---|
| MS MARCO Passage | Web passages | 8.84M passages | In-domain training & eval |
| MS MARCO Document | Web docs | 3.2M docs | In-domain generation eval |
| Natural Questions | Wikipedia QA | 100 val queries | Out-of-domain eval |
| TREC Robust04 | Newswire | 249 topics | Out-of-domain eval |

Download instructions: see [`data/README.md`](data/README.md)

---

## 🚀 Quick Start

### Run the full pipeline
```bash
python run_pipeline.py \
  --dataset msmarco_passage \
  --retriever dense \
  --model random_forest \
  --top_k 20
```

### Run only QPP feature extraction + prediction
```bash
python run_pipeline.py \
  --dataset msmarco_passage \
  --retriever dense \
  --mode qpp_only
```

### Run QPP-guided adaptive RAG
```bash
python run_pipeline.py \
  --dataset msmarco_document \
  --retriever dense \
  --generator bart \
  --mode adaptive_rag
```

---

## 🧩 12-Dimensional QPP Feature Set

| # | Feature | Category | Description |
|---|---|---|---|
| 1 | `max_sim` | Semantic | Max cosine similarity: query vs any top-20 passage |
| 2 | `sim_variance` | Semantic | Variance of cosine similarity scores |
| 3 | `high_sim_count` | Semantic | Count of passages with similarity > 0.8 |
| 4 | `rank_dropoff` | Semantic | Similarity drop from rank-1 to rank-10 |
| 5 | `emb_variance` | Embedding | Avg squared distance of passage embeddings from centroid |
| 6 | `term_overlap` | Lexical | Avg fraction of query terms in each retrieved passage |
| 7 | `query_length` | Lexical | Token count of the query |
| 8 | `query_idf_sum` | Lexical | Sum of IDF scores for query terms |
| 9 | `query_entropy` | Lexical | Shannon entropy of IDF-weighted query terms |
| 10 | `clarity` | Traditional QPP | KL-divergence: passage LM vs corpus LM |
| 11 | `wig` | Traditional QPP | Weighted Information Gain (top-5 vs rank-6) |
| 12 | `nqc` | Traditional QPP | Normalized Query Commitment (score dispersion) |

---

## 🔬 Retrieval Architectures

| Retriever | Type | Model/Library |
|---|---|---|
| BM25 | Sparse | Pyserini (k1=0.9, b=0.4) |
| Sentence-BERT | Dense | `msmarco-distilbert-base-v4` + FAISS |
| Hybrid | BM25 + Dense | Linear fusion (α=0.5) |
| ColBERT | Late-interaction | `colbert-ir/colbertv2.0` + PLAID index |

---

## 📐 QPP-Guided Adaptive Retrieval

Queries are stratified into 3 difficulty tiers:

```
Low-QPP  (ŷ < τ₁)  → k = 50  (deeper retrieval)
Mid-QPP  (τ₁ ≤ ŷ < τ₂) → k = 30
High-QPP (ŷ ≥ τ₂)  → k = 20  (standard retrieval)
```

Where τ₁ = 33rd percentile and τ₂ = 66th percentile of predicted scores.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{sinha2026ragqpp,
  title     = {Adaptive Query Performance Prediction for Retrieval-Augmented Generation:
               Bridging Retrieval Quality and Generation Relevance},
  author    = {Sinha, Aparajita and Chakma, Kunal},
  journal   = {ACM Transactions on Information Systems},
  year      = {2026},
  note      = {Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Pyserini](https://github.com/castorini/pyserini) for BM25 retrieval
- [Sentence-Transformers](https://www.sbert.net/) for dense encoding
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) for late-interaction retrieval
- [HuggingFace](https://huggingface.co/) for BART-large and LLaMA-3-8B
