"""Fine-tuned sentence embeddings; cosine is not a probability."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from src.models.runtime import checkpoint_key, load_bundle, encode_texts, validate_pairs, batches
from src.models.tang3_triplet_encoder.model import BiEncoder

_cache = {}


def load_cached_model(checkpoint_dir, pretrained_name=None, use_cache=True, device=None):
    key = checkpoint_key(checkpoint_dir, 3, device) + (pretrained_name,)
    if use_cache and key in _cache:
        return _cache[key]
    model, tokenizer, meta = load_bundle(checkpoint_dir, 3, BiEncoder, device)
    if pretrained_name is not None and pretrained_name != meta['pretrained_name']:
        raise ValueError('Backbone override disagrees with checkpoint metadata')
    result = model, tokenizer
    if use_cache:
        _cache.clear()
        _cache[key] = result
    return result


def tang3_predict_batch(pairs, checkpoint_dir='checkpoints/tang3', batch_size=32,
                        model=None, tokenizer=None):
    pairs = validate_pairs(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not pairs:
        return pd.DataFrame(columns=['sentence_a', 'sentence_b', 'similarity', 'needs_tang4'])
    if model is None or tokenizer is None:
        model, tokenizer = load_cached_model(checkpoint_dir)
    length = getattr(model, '_inference_metadata', {}).get('max_length', 128)
    scores = []
    for batch in batches(pairs, batch_size):
        a = encode_texts([p[0] for p in batch], model, tokenizer, length, batch_size)
        b = encode_texts([p[1] for p in batch], model, tokenizer, length, batch_size)
        scores.extend((a * b).sum(dim=1).clamp(-1, 1).tolist())
    return pd.DataFrame({'sentence_a': [p[0] for p in pairs], 'sentence_b': [p[1] for p in pairs],
        'similarity': scores, 'needs_tang4': True})


def tang3_predict(sentence_a, sentence_b, checkpoint_dir='checkpoints/tang3', model=None, tokenizer=None):
    row = tang3_predict_batch([(sentence_a, sentence_b)], checkpoint_dir, 1, model, tokenizer).iloc[0]
    return {'similarity': float(row.similarity), 'score_type': 'cosine',
            'decision': 'continue', 'needs_tang4': True}
