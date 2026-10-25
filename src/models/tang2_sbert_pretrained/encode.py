"""Pretrained sentence embeddings. Scores are features, never early-exit labels."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
import torch
from transformers import AutoTokenizer
from src.models.runtime import EMBEDDING_MODEL, encode_texts, validate_pairs, batches
from src.models.tang3_triplet_encoder.model import BiEncoder

DEFAULT_MODEL_NAME = EMBEDDING_MODEL
_cache = {}


def load_model(model_name=DEFAULT_MODEL_NAME, use_cache=True, device=None):
    device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    key = model_name, device
    if use_cache and key in _cache:
        return _cache[key]
    result = BiEncoder(model_name).to(device).eval(), AutoTokenizer.from_pretrained(model_name)
    if use_cache:
        _cache.clear()
        _cache[key] = result
    return result


def _encode_batch(sentences, model, tokenizer, max_length=128, batch_size=32):
    return encode_texts(sentences, model, tokenizer, max_length, batch_size)


def sbert_filter_batch(pairs, model=None, tokenizer=None, low_threshold=None,
                       high_threshold=None, batch_size=32, max_length=128):
    """Legacy threshold arguments are accepted but never suppress later tiers."""
    pairs = validate_pairs(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not pairs:
        return pd.DataFrame(columns=['sentence_a', 'sentence_b', 'similarity', 'decision'])
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    scores = []
    for batch in batches(pairs, batch_size):
        a = encode_texts([p[0] for p in batch], model, tokenizer, max_length, batch_size)
        b = encode_texts([p[1] for p in batch], model, tokenizer, max_length, batch_size)
        scores.extend((a * b).sum(dim=1).clamp(-1, 1).tolist())
    return pd.DataFrame({'sentence_a': [p[0] for p in pairs], 'sentence_b': [p[1] for p in pairs],
                         'similarity': scores, 'decision': 'continue'})


def sbert_filter(sentence_a, sentence_b, model=None, tokenizer=None,
                 low_threshold=None, high_threshold=None):
    row = sbert_filter_batch([(sentence_a, sentence_b)], model, tokenizer, batch_size=1).iloc[0]
    return {'similarity': float(row.similarity), 'decision': 'continue', 'score_type': 'cosine', 'reason': None}
