# RAG-QPP

Code for the paper **"Adaptive Query Performance Prediction for Retrieval-Augmented Generation: Bridging Retrieval Quality and Generation Relevance"**

> Aparajita Sinha & Kunal Chakma — National Institute of Technology Agartala, India

---

## What This Does

RAG-QPP predicts query difficulty in RAG pipelines using a 12-dimensional post-retrieval feature set — without needing relevance judgments or generator-internal signals. The predicted score is used to adaptively adjust retrieval depth for better answer quality.

---

## Repository Structure

```
rag-qpp/
├── src/
│   ├── features.py       # 12-D QPP feature extraction
│   ├── retrieval.py      # BM25, Dense, Hybrid, ColBERT
│   ├── models.py         # Random Forest, XGBoost, LightGBM
│   ├── generation.py     # BART-large, LLaMA-3-8B
│   ├── evaluate.py       # All metrics
│   └── adaptive_rag.py   # QPP-guided adaptive retrieval
├── notebooks/            # Experiments (RQ1–RQ5)
├── configs/config.yaml   # Hyperparameters
├── run_pipeline.py       # End-to-end runner
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-qpp.git
cd rag-qpp
pip install -r requirements.txt
```

---

## Usage

```bash
# QPP prediction only
python run_pipeline.py --dataset msmarco_passage --retriever dense --mode qpp_only

# Full adaptive RAG pipeline
python run_pipeline.py --dataset msmarco_document --retriever dense --generator bart --mode adaptive_rag

# Feature ablation study
python run_pipeline.py --dataset msmarco_passage --retriever dense --mode ablation
```

---

## Datasets

| Dataset | Download Link |
|---|---|
| MS MARCO Passage | [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/) |
| MS MARCO Document | [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/) |
| Natural Questions | [https://huggingface.co/datasets/google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open) |
| TREC Robust04 | [https://ir.nist.gov/data/cd45-nocr.html](https://ir.nist.gov/data/cd45-nocr.html) |
