"""Shared checkpoint, batching and reproducibility utilities for trainable tiers."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FORMAT_VERSION = 1


def make_encoder(pretrained_name, encoder_config=None):
    if encoder_config is None:
        return AutoModel.from_pretrained(pretrained_name)
    config = dict(encoder_config)
    model_type = config.pop('model_type')
    return AutoModel.from_config(AutoConfig.for_model(model_type, **config))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_threshold(probs, labels):
    """Maximum positive-class F1 on validation only; vectorized exact thresholds."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def save_bundle(model, tokenizer, directory, tier, pretrained_name, max_length,
                threshold=None, seed=42, **training):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {'format_version': FORMAT_VERSION, 'tier': tier,
            'pretrained_name': pretrained_name, 'max_length': int(max_length),
            'threshold': threshold, 'seed': seed,
            'encoder_config': model.encoder.config.to_dict(),
            'dropout': getattr(getattr(model, 'dropout', None), 'p', 0.0),
            'training': training,
            'label_semantics': 'AI-touched' if tier == 0 else 'paraphrase_relation'}
    # JSON round-trip ensures weights_only loading only needs primitive metadata.
    meta = json.loads(json.dumps(meta))
    tokenizer.save_pretrained(directory)
    path = directory / f'tang{tier}_best.pt'
    temporary = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': model.state_dict(), 'metadata': meta}, temporary)
    temporary.replace(path)
    (directory / 'model_config.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    model._inference_metadata = meta


def read_bundle(directory, tier):
    path = Path(directory) / f'tang{tier}_best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'metadata' not in payload or 'state_dict' not in payload:
        raise ValueError(f'{path}: legacy checkpoint has no configuration. Retrain with the new train.py.')
    meta = payload['metadata']
    if meta.get('format_version') != FORMAT_VERSION or meta.get('tier') != tier:
        raise ValueError(f'Unsupported checkpoint format or wrong tier: {path}')
    return payload


def load_bundle(directory, tier, model_class, device=None):
    payload = read_bundle(directory, tier)
    meta = payload['metadata']
    kwargs = {'pretrained_name': meta['pretrained_name'], 'encoder_config': meta['encoder_config']}
    if tier != 3:
        kwargs['dropout'] = meta['dropout']
    model = model_class(**kwargs)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu')).eval()
    model._inference_metadata = meta
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model, tokenizer, meta


def checkpoint_key(directory, tier, device=None):
    path = (Path(directory) / f'tang{tier}_best.pt').resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size, str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


def restore_best(model, directory, tier):
    payload = read_bundle(directory, tier)
    model.load_state_dict(payload['state_dict'])
    model._inference_metadata = payload['metadata']
    model.eval()
    return payload['metadata']


def validate_texts(texts):
    texts = list(texts)
    if any(not isinstance(s, str) or not s.strip() for s in texts):
        raise ValueError('Expected non-empty text strings; empty input is not a negative prediction.')
    return texts


def validate_pairs(pairs):
    pairs = list(pairs)
    if any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in pairs):
        raise ValueError('Each pair must contain two sentences')
    validate_texts([s for p in pairs for s in p])
    return pairs


def batches(items, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@torch.inference_mode()
def encode_texts(texts, model, tokenizer, max_length=128, batch_size=32):
    texts = validate_texts(texts)
    model.eval()
    device = next(model.parameters()).device
    result = []
    for batch in batches(texts, batch_size):
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
        result.append(model(enc['input_ids'].to(device), enc['attention_mask'].to(device)).cpu())
    return torch.cat(result) if result else torch.empty((0, model.encoder.config.hidden_size))
