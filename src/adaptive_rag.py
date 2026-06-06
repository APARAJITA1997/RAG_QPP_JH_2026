# src/adaptive_rag.py
# QPP-guided adaptive RAG pipeline with two-component decision per query:
#   k(q) -> retrieval depth
#   lambda(q) -> how much the retrieved context should shape the answer

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# -- adjustable bounds for retrieval depth --
K_FLOOR = 5
K_CEIL  = 20

# sigmoid steepness -- controls how sharply lambda responds to QPP score changes
SIGMOID_ALPHA = 6.0

# lambda cutoffs separating the three prompt modes
HIGH_CUT = 0.7
LOW_CUT  = 0.4


@dataclass
class QueryResult:
    query:       str
    qpp_score:   float
    k_used:      int
    lambda_used: float   # context-influence weight, derived from QPP score
    gen_mode:    str     # RETRIEVAL_PRIMARY / BALANCED / KNOWLEDGE_PRIMARY
    answer:      str
    rouge_l:     float = 0.0
    bert_f1:     float = 0.0
    token_f1:    float = 0.0


def score_to_lambda(qpp_score, alpha=SIGMOID_ALPHA):
    # sigmoid centred at 0.5:
    #   qpp_score near 1 -> lambda near 1  (retrieval trusted, use it heavily)
    #   qpp_score near 0 -> lambda near 0  (retrieval unreliable, lean on model)
    return 1.0 / (1.0 + math.exp(-alpha * (qpp_score - 0.5)))


def score_to_k(qpp_score, lo=K_FLOOR, hi=K_CEIL):
    # linearly interpolate between depth bounds based on QPP score
    return lo + round((hi - lo) * float(qpp_score))


def get_gen_mode(lam):
    if lam > HIGH_CUT:
        return "RETRIEVAL_PRIMARY"
    elif lam > LOW_CUT:
        return "BALANCED"
    return "KNOWLEDGE_PRIMARY"


def make_prompt(query, context, lambda_q):
    """
    Build the generation prompt conditioned on lambda_q.

    This is the concrete mechanism by which lambda(q) influences generation:
    the instruction text sent to the generator changes depending on
    how reliable the QPP model estimates the retrieval to be.

      lambda_q > 0.7  -> passages are treated as the primary source
      lambda_q > 0.4  -> passages used alongside model's own knowledge
      lambda_q <= 0.4 -> model answers from parametric knowledge;
                         passages treated as optional background only
    """
    if lambda_q > 0.7:
        prompt = (
            f"Using the following passages as your primary source:\n"
            f"{context}\n\n"
            f"Answer: {query}"
        )
    elif lambda_q > 0.4:
        prompt = (
            f"Consider these passages, but also use your own knowledge:\n"
            f"{context}\n\n"
            f"Answer: {query}"
        )
    else:
        prompt = (
            f"Answer from your own knowledge. "
            f"Additional context (use sparingly):\n"
            f"{context}\n\n"
            f"Answer: {query}"
        )
    return prompt


class AdaptiveRAGPipeline:
    """
    End-to-end QPP-guided adaptive RAG.

    Per query:
      1. Run QPP model -> score s
      2. Compute k(q) = score_to_k(s)  and  lambda(q) = score_to_lambda(s)
      3. Retrieve top-k(q) passages
      4. Build lambda-conditioned prompt via make_prompt()
      5. Generate answer and score against reference if available
    """

    def __init__(self, retriever, generator, qpp_model,
                 scaler=None, df_lookup=None, corpus_lm=None,
                 query_emb_fn=None, k_lo=K_FLOOR, k_hi=K_CEIL):

        self.retriever    = retriever
        self.generator    = generator
        self.qpp_model    = qpp_model
        self.scaler       = scaler
        self.df_lookup    = df_lookup or {}
        self.corpus_lm    = corpus_lm or {}
        self.query_emb_fn = query_emb_fn
        self.k_lo         = k_lo
        self.k_hi         = k_hi

    def _get_qpp_score(self, query):
        from src.features import extract_features_batch

        # pull max passages so features reflect the full candidate set
        hits  = self.retriever.retrieve(query, top_k=self.k_hi)
        texts = [h["text"] for h in hits]
        embs  = np.stack([h["embedding"] for h in hits])
        q_vec = (
            self.query_emb_fn(query)
            if self.query_emb_fn is not None
            else np.zeros(768, dtype=np.float32)
        )

        X = extract_features_batch(
            [query], [texts], [embs],
            q_vec[np.newaxis, :],
            self.df_lookup, self.corpus_lm
        )
        if self.scaler is not None:
            X = self.scaler.transform(X)

        raw = float(self.qpp_model.predict(X)[0])
        return float(np.clip(raw, 0.0, 1.0))

    def _score_answer(self, pred, ref):
        rl = bf = tf = 0.0

        try:
            from rouge_score import rouge_scorer
            rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)\
                             .score(ref, pred)["rougeL"].fmeasure
        except Exception:
            pass

        try:
            from bert_score import score as bs
            _, _, F = bs([pred], [ref], lang="en", verbose=False)
            bf = float(F[0])
        except Exception:
            pass

        p_tok = set(pred.lower().split())
        r_tok = set(ref.lower().split())
        if p_tok and r_tok:
            common = p_tok & r_tok
            pr = len(common) / len(p_tok)
            rc = len(common) / len(r_tok)
            if pr + rc > 0:
                tf = 2 * pr * rc / (pr + rc)

        return {"rouge_l": rl, "bert_f1": bf, "token_f1": tf}

    def run(self, queries: List[str],
            references: Optional[List[str]] = None) -> List[QueryResult]:

        out = []

        for idx, q in enumerate(queries):

            # step 1: QPP score
            s = self._get_qpp_score(q)

            # step 2: adaptive policy A(q) = (k(q), lambda(q))
            k      = score_to_k(s, self.k_lo, self.k_hi)
            lam    = score_to_lambda(s)
            mode   = get_gen_mode(lam)

            log.info("[%d/%d] QPP=%.4f  k=%d  lambda=%.4f  mode=%s",
                     idx + 1, len(queries), s, k, lam, mode)

            # step 3: retrieve top-k(q) passages
            hits    = self.retriever.retrieve(q, top_k=k)
            context = "\n\n".join(h["text"] for h in hits)

            # step 4: build lambda-conditioned prompt
            prompt = make_prompt(q, context, lam)

            # step 5: generate
            answer = self.generator.generate(prompt)

            # step 6: evaluate if reference available
            metrics = {"rouge_l": 0.0, "bert_f1": 0.0, "token_f1": 0.0}
            if references and idx < len(references) and references[idx]:
                metrics = self._score_answer(answer, references[idx])

            out.append(QueryResult(
                query       = q,
                qpp_score   = s,
                k_used      = k,
                lambda_used = lam,
                gen_mode    = mode,
                answer      = answer,
                **metrics
            ))

        return out
