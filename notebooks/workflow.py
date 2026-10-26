"""Shared notebook utilities. No downloads, training or API calls on import."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd


def inspect_csv(path, text_columns, sample_size=5000, seed=42):
    """Exact streamed counts plus uniform bounded sample (random-priority reservoir)."""
    rng=np.random.default_rng(seed);sample=pd.DataFrame();labels=Counter();sources=Counter()
    count=0;missing=0;bad_labels=0
    for chunk in pd.read_csv(path,chunksize=50000,dtype={'group_id':str,'candidate_group_id':str,'negative_group_id':str}):
        if not set(text_columns).issubset(chunk):raise ValueError(f'Missing text columns: {path}')
        count+=len(chunk)
        missing+=int(chunk[text_columns].isna().any(axis=1).sum())
        if 'label' in chunk:
            labels.update(chunk.label.value_counts().to_dict());bad_labels+=int((~chunk.label.isin([0,1])).sum())
        if 'source' in chunk:sources.update(chunk.source.value_counts().to_dict())
        chunk=chunk.assign(_sample_priority=rng.random(len(chunk)))
        sample=pd.concat([sample,chunk],ignore_index=True).nsmallest(sample_size,'_sample_priority')
    if not count:raise ValueError(f'Empty CSV: {path}')
    if missing or bad_labels:raise ValueError(f'{path}: missing text rows={missing}, invalid labels={bad_labels}')
    sample=sample.drop(columns='_sample_priority').reset_index(drop=True)
    if sample[text_columns].apply(lambda col:col.str.strip().eq('')).any().any():raise ValueError(f'Empty sampled text: {path}')
    return {'file':Path(path).name,'rows':count,'labels':dict(labels),'sources':dict(sources)},sample


def run_script(root, relative_path, arguments, enabled=False, log_path=None):
    command=[sys.executable,'-X','utf8',str(Path(root)/relative_path),*[str(a) for a in arguments]]
    print('Command arguments:',command)
    if not enabled:
        print('Chưa chạy. Bật cờ RUN_* trong ô cấu hình khi sẵn sàng.');return
    import os
    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
    handle=None
    try:
        if log_path:
            Path(log_path).parent.mkdir(parents=True,exist_ok=True)
            handle=Path(log_path).open('w',encoding='utf-8')
        with subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace') as process:
            for line in process.stdout:
                print(line,end='')
                if handle:handle.write(line);handle.flush()
            if process.wait():raise RuntimeError(f'Script failed; inspect log: {log_path}')
    finally:
        if handle:handle.close()


def read_json(path):
    path=Path(path)
    if not path.is_file():
        print(f'Chưa có artifact: {path}');return None
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require_checkpoint(directory,tier):
    directory=Path(directory)
    for name in (f'tang{tier}_best.pt','model_config.json','tokenizer_config.json'):
        path=directory/name
        if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(f'Missing trained artifact: {path}')
    return directory
