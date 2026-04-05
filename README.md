# RAG-QPP

**Adaptive Query Performance Prediction for Retrieval-Augmented Generation: Bridging Retrieval Quality and Generation Relevance**

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
git clone https://github.com/APARAJITA1997/RAG_QPP_JH_2026.git
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

---

## Citation

If you use our code or find our work helpful, please cite the following paper:

```bibtex
@article{sinha2026ragqpp,
  author    = {Aparajita Sinha and Kunal Chakma},
  title     = {Adaptive Query Performance Prediction for Retrieval-Augmented Generation: 
               Bridging Retrieval Quality and Generation Relevance},
  journal   = {ACM Transactions on Information Systems},
  year      = {2026},
  pages     = {1--54},
  doi       = {XXXXXXX.XXXXXXX},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

> **Please cite the paper if you are using our RAG-QPP framework in your research. Your citations help support our work and encourage further development.**

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or feedback, please contact:
- Aparajita Sinha — aparajitas824@gmail.com
- Kunal Chakma — kchakma.cse@nita.ac.in

National Institute of Technology Agartala, Tripura, India
