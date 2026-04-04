"""
scripts/build_corpus_stats.py
------------------------------
Pre-compute corpus statistics required by the QPP feature extractor:

  1. df_lookup.json  — {term: document_frequency}
     Used by: query_idf_sum (Feature 8), query_entropy (Feature 9)

  2. corpus_lm.json  — {word: P(w|C)}  unigram language model
     Used by: Clarity score (Feature 10)

Run once per corpus before training QPP models.

Usage:
    python scripts/build_corpus_stats.py \\
        --corpus data/msmarco_passage/collection.tsv \\
        --output_dir data/msmarco_passage/ \\
        --max_docs 8841823

The corpus TSV is expected to have format:  pid\\tpassage_text
(standard MS MARCO collection format).

Reference: Sinha & Chakma (2026), Section 3.13 (Features 8-10)
"""

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Simple tokenizer (mirrors src/features.py) ────────────────────────────────

STOPWORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'and', 'or', 'but', 'not', 'with', 'this', 'that', 'are',
    'was', 'be', 'by', 'as', 'from', 'which', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'its', 'their', 'they', 'we', 'you', 'he', 'she', 'i', 'me',
}

def tokenize(text: str, remove_stopwords: bool = False):
    """Lowercase, strip punctuation, optionally remove stopwords."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Build corpus statistics for RAG-QPP.')
    p.add_argument('--corpus',      required=True,
                   help='Path to corpus TSV file (pid\\ttext per line).')
    p.add_argument('--output_dir',  required=True,
                   help='Directory to write df_lookup.json and corpus_lm.json.')
    p.add_argument('--max_docs',    type=int, default=None,
                   help='Limit number of docs (useful for testing).')
    p.add_argument('--min_df',      type=int, default=2,
                   help='Minimum document frequency to include a term.')
    p.add_argument('--remove_stopwords', action='store_true',
                   help='Exclude stopwords from statistics.')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_counter: Counter  = Counter()   # {term: num docs containing term}
    tf_total:   Counter  = Counter()   # {word: total occurrences} for LM
    n_docs = 0
    n_tokens_total = 0

    logger.info('Reading corpus from %s …', args.corpus)
    with open(args.corpus, encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, desc='Processing documents', unit=' docs'):
            if args.max_docs and n_docs >= args.max_docs:
                break

            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) < 2:
                continue

            text   = parts[1]
            tokens = tokenize(text, args.remove_stopwords)
            if not tokens:
                continue

            # Document frequency: count each UNIQUE term once per doc
            unique_terms = set(tokens)
            for t in unique_terms:
                df_counter[t] += 1

            # Total frequency for corpus LM
            for t in tokens:
                tf_total[t] += 1

            n_tokens_total += len(tokens)
            n_docs         += 1

    logger.info('Processed %d documents, %d total tokens.', n_docs, n_tokens_total)
    logger.info('Unique vocabulary size: %d', len(df_counter))

    # ── Filter rare terms ─────────────────────────────────────────────────────
    df_filtered = {
        term: cnt
        for term, cnt in df_counter.items()
        if cnt >= args.min_df
    }
    logger.info(
        'After min_df=%d filter: %d terms retained.',
        args.min_df, len(df_filtered)
    )

    # ── Corpus language model  P(w|C) ─────────────────────────────────────────
    # Use only terms that passed the min_df filter
    total_kept = sum(tf_total[t] for t in df_filtered)
    corpus_lm  = {
        term: tf_total[term] / total_kept
        for term in df_filtered
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    df_path = out_dir / 'df_lookup.json'
    lm_path = out_dir / 'corpus_lm.json'

    with open(df_path, 'w') as f:
        json.dump(df_filtered, f, separators=(',', ':'))
    logger.info('df_lookup saved → %s  (%d terms)', df_path, len(df_filtered))

    with open(lm_path, 'w') as f:
        json.dump(corpus_lm, f, separators=(',', ':'))
    logger.info('corpus_lm saved → %s  (%d terms)', lm_path, len(corpus_lm))

    # ── Sanity checks ─────────────────────────────────────────────────────────
    N = n_docs
    sample_terms = ['machine', 'learning', 'deep', 'neural', 'network']
    logger.info('\nSample IDF values (log(N / df + 1)):')
    for t in sample_terms:
        df = df_filtered.get(t, 0)
        idf = math.log(N / (df + 1))
        lm  = corpus_lm.get(t, 0.0)
        logger.info('  %-12s  df=%8d  idf=%.4f  P(w|C)=%.6f', t, df, idf, lm)

    logger.info('Done.')


if __name__ == '__main__':
    main()
