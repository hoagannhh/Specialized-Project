"""Predict a paraphrase relationship, not a verdict about plagiarism/authorship."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
import torch
from src.models.runtime import checkpoint_key, load_bundle, validate_pairs, batches
from src.models.tang4_cross_encoder.model import CrossEncoderModel

_cache = {}


def load_cached_model(checkpoint_dir, pretrained_name=None, use_cache=True, device=None):
    key = checkpoint_key(checkpoint_dir, 4, device) + (pretrained_name,)
    if use_cache and key in _cache:
        return _cache[key]
    model, tokenizer, meta = load_bundle(checkpoint_dir, 4, CrossEncoderModel, device)
    if pretrained_name is not None and pretrained_name != meta['pretrained_name']:
        raise ValueError('Backbone override disagrees with checkpoint metadata')
    result = model, tokenizer
    if use_cache:
        _cache.clear()
        _cache[key] = result
    return result


@torch.inference_mode()
def tang4_predict_batch(pairs, checkpoint_dir='checkpoints/tang4', threshold=None, batch_size=32):
    pairs = validate_pairs(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not pairs:
        return pd.DataFrame(columns=['sentence_a', 'sentence_b', 'probability', 'label', 'threshold_used'])
    model, tokenizer = load_cached_model(checkpoint_dir)
    meta = model._inference_metadata
    threshold = meta['threshold'] if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must lie in [0,1]')
    device = next(model.parameters()).device
    probabilities = []
    for batch in batches(pairs, batch_size):
        enc = tokenizer([p[0] for p in batch], [p[1] for p in batch], truncation=True,
                        max_length=meta['max_length'], padding=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in enc.items() if k in ('input_ids','attention_mask','token_type_ids')}
        probabilities.extend(model.predict_proba(**inputs).cpu().tolist())
    return pd.DataFrame({'sentence_a': [p[0] for p in pairs], 'sentence_b': [p[1] for p in pairs],
        'probability': probabilities, 'threshold_used': threshold,
        'label': ['nghi vấn paraphrase' if p >= threshold else 'không phát hiện quan hệ paraphrase' for p in probabilities]})


def tang4_predict(sentence_a, sentence_b, checkpoint_dir='checkpoints/tang4', threshold=None):
    row = tang4_predict_batch([(sentence_a, sentence_b)], checkpoint_dir, threshold, 1).iloc[0]
    return {'probability': float(row.probability), 'label': row.label,
            'threshold_used': float(row.threshold_used), 'probability_calibrated': False,
            'target': 'paraphrase_relation'}
