"""
src/generation.py
-----------------
Answer generation using two complementary generative architectures:
  1. BART-large  — encoder-decoder sequence-to-sequence (facebook/bart-large)
  2. LLaMA-3-8B  — decoder-only large language model (meta-llama/Llama-3-8B)

Both models are conditioned on the same prompt template (Eq. 50):
    "Question: {query}  Context: {top-5 passages}"

Input is capped at 512 tokens; output is capped at 50 (BART) / 80 (LLaMA).
The top-5 passages from the retriever are concatenated as context.

Reference: Sinha & Chakma (2026), Sections 3.12, 4
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS  = 512
BART_MAX_NEW_TOKENS = 50
LLAMA_MAX_NEW_TOKENS = 80
TOP_P_LLAMA = 0.9          # nucleus sampling p (Section 4)
BEAM_WIDTH  = 5            # BART beam search width (Section 4)
CONTEXT_PASSAGES = 5       # top-k passages passed to generator (Section 4)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder  (Eq. 50)
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str, passages: List[str], factual: bool = True) -> str:
    """
    Construct the generation prompt.

    In the QPP-guided setting the prompt includes an instruction to rely
    only on the retrieved passages (controlled prompting, Section 5.3).
    """
    context = " ".join(passages[:CONTEXT_PASSAGES])
    if factual:
        return (
            "Using only the information from the retrieved passages, "
            "generate a concise and accurate answer to the query. "
            "Do not include unsupported content. "
            f"Question: {query} Context: {context}"
        )
    return f"Question: {query} Context: {context}"


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, passages: List[str]) -> str:
        """Return a single answer string."""

    def batch_generate(
        self, queries: List[str], passages_list: List[List[str]]
    ) -> List[str]:
        return [self.generate(q, p) for q, p in zip(queries, passages_list)]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BART-large  (encoder-decoder, beam search)
# ══════════════════════════════════════════════════════════════════════════════

class BARTGenerator(BaseGenerator):
    """
    BART-large encoder-decoder generator.
    Checkpoint: facebook/bart-large
    Decoding:   beam search (num_beams=5), max 50 new tokens.

    Eq. 51:  a_gen = argmax_a  P(a | x; θ_BART)
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        device: Optional[str] = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_new_tokens: int = BART_MAX_NEW_TOKENS,
        num_beams: int = BEAM_WIDTH,
    ):
        try:
            import torch
            from transformers import BartForConditionalGeneration, BartTokenizer
        except ImportError:
            raise ImportError("transformers and torch are required: pip install transformers torch")

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens   = max_new_tokens
        self.num_beams        = num_beams

        logger.info("Loading BART-large (%s) on %s …", model_name, self.device)
        from transformers import BartForConditionalGeneration, BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model     = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BART-large ready.")

    def generate(self, query: str, passages: List[str]) -> str:
        import torch
        prompt = build_prompt(query, passages, factual=True)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LLaMA-3-8B  (decoder-only, nucleus sampling)
# ══════════════════════════════════════════════════════════════════════════════

class LLaMAGenerator(BaseGenerator):
    """
    LLaMA-3-8B decoder-only generator.
    Checkpoint: meta-llama/Llama-3-8B  (or any compatible variant).
    Decoding:   nucleus sampling (top_p=0.9), max 80 new tokens.

    Eq. 52:  P(a | x; θ_LLaMA) = Π_{j} P(t_j | x, t_{1:j-1}; θ_LLaMA)

    Note: You need a HuggingFace access token for gated LLaMA checkpoints.
          Set the HF_TOKEN environment variable or pass `hf_token`.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_new_tokens: int = LLAMA_MAX_NEW_TOKENS,
        top_p: float = TOP_P_LLAMA,
        load_in_8bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers and torch are required: pip install transformers torch")

        import torch
        import os
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens   = max_new_tokens
        self.top_p            = top_p

        token = hf_token or os.environ.get("HF_TOKEN")
        logger.info("Loading LLaMA-3-8B (%s) on %s …", model_name, self.device)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"token": token} if token else {}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("LLaMA-3-8B ready.")

    def generate(self, query: str, passages: List[str]) -> str:
        import torch
        prompt  = build_prompt(query, passages, factual=True)
        inputs  = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Strip the prompt tokens; return only the generated part
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_generator(generator_type: str, **kwargs) -> BaseGenerator:
    """
    Instantiate a generator by name.

    Parameters
    ----------
    generator_type : "bart" | "llama"
    **kwargs       : forwarded to the generator's __init__.
    """
    mapping = {
        "bart":  BARTGenerator,
        "llama": LLaMAGenerator,
    }
    key = generator_type.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown generator '{generator_type}'. Choose from {list(mapping)}"
        )
    return mapping[key](**kwargs)
