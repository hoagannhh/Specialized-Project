from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
import torch
from src.models.runtime import checkpoint_key, load_bundle, validate_texts, batches
from src.models.tang0_style_classifier.model import SingleSentenceClassifier

_cache = {}


def load_for_inference(checkpoint_dir, pretrained_name=None, use_cache=True, device=None):
    key = checkpoint_key(checkpoint_dir, 0, device) + (pretrained_name,)
    if use_cache and key in _cache:
        return _cache[key]
    model, tokenizer, meta = load_bundle(checkpoint_dir, 0, SingleSentenceClassifier, device)
    if pretrained_name is not None and pretrained_name != meta['pretrained_name']:
        raise ValueError('Backbone override disagrees with checkpoint metadata')
    result = model, tokenizer, meta['threshold']
    if use_cache:
        _cache.clear()
        _cache[key] = result
    return result


@torch.inference_mode()
def predict_batch(sentences, checkpoint_dir='checkpoints/tang0', max_length=None, batch_size=32):
    sentences = validate_texts(sentences)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    columns = ['sentence', 'probability', 'decision', 'threshold_used']
    if not sentences:
        return pd.DataFrame(columns=columns)
    model, tokenizer, threshold = load_for_inference(checkpoint_dir)
    length = model._inference_metadata['max_length'] if max_length is None else max_length
    device = next(model.parameters()).device
    values = []
    for batch in batches(sentences, batch_size):
        encoded = tokenizer(batch, truncation=True, max_length=length, padding=True, return_tensors='pt')
        probs = model.predict_proba(encoded['input_ids'].to(device), encoded['attention_mask'].to(device))
        values.extend(probs[:, 1].cpu().tolist())
    return pd.DataFrame({'sentence': sentences, 'probability': values,
        'decision': ['suspicious' if p >= threshold else 'skip' for p in values],
        'threshold_used': threshold})


def predict_sentence(sentence, checkpoint_dir='checkpoints/tang0', max_length=None):
    row = predict_batch([sentence], checkpoint_dir, max_length, batch_size=1).iloc[0]
    return {'probability': float(row.probability), 'decision': row.decision,
            'threshold_used': float(row.threshold_used), 'probability_calibrated': False,
            'target': 'AI-touched'}
