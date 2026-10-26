"""CPU-only EDA helpers; exact streaming counts, bounded samples, optional disk audit."""
from collections import Counter
from pathlib import Path
import hashlib
import sqlite3
import unicodedata
import numpy as np
import pandas as pd


def normalize_text(value):
    return ' '.join(unicodedata.normalize('NFC',str(value)).split())


def profile_csv(path, text_columns, sample_size=5000, seed=42):
    if sample_size<1:raise ValueError('sample_size must be positive')
    rng=np.random.default_rng(seed);sample=pd.DataFrame();counts=Counter();categories={};source_labels=Counter()
    missing=Counter();empty=Counter();length_totals=Counter();length_hist={c:Counter() for c in text_columns}
    for chunk in pd.read_csv(path,chunksize=50000,dtype={'group_id':str,'candidate_group_id':str,'negative_group_id':str}):
        if not set(text_columns).issubset(chunk):raise ValueError(f'Missing text columns in {path}')
        counts['rows']+=len(chunk)
        for col in text_columns:
            missing[col]+=int(chunk[col].isna().sum())
            text=chunk[col].fillna('').astype(str)
            empty[col]+=int(text.str.strip().eq('').sum())
            lengths=text.str.split().str.len();length_totals[col]+=int(lengths.sum())
            length_hist[col].update(lengths.value_counts().to_dict())
        for col in ('label','source','negative_type','pair_type','original_task_label','sentence_position'):
            if col in chunk:categories.setdefault(col,Counter()).update(chunk[col].fillna('<MISSING>').astype(str).value_counts().to_dict())
        if 'label' in chunk:counts['invalid_labels']+=int((~chunk.label.isin([0,1])).sum())
        if {'source','label'}.issubset(chunk):
            source_labels.update(chunk.assign(source=chunk.source.fillna('<MISSING>')).groupby(['source','label'],dropna=False).size().to_dict())
        if {'group_id','negative_group_id'}.issubset(chunk):
            counts['invalid_negative_group']+=int(chunk.group_id.eq(chunk.negative_group_id).sum())
            negative=chunk.negative.map(normalize_text)
            counts['negative_equals_anchor_or_positive']+=int((negative.eq(chunk.anchor.map(normalize_text)) | negative.eq(chunk.positive.map(normalize_text))).sum())
        if {'group_id','candidate_group_id','label'}.issubset(chunk):
            counts['invalid_pair_provenance']+=int(chunk.group_id.eq(chunk.candidate_group_id).ne(chunk.label.eq(1)).sum())
            counts['identical_negative_pair']+=int((chunk.label.eq(0)&chunk.sentence_1.map(normalize_text).eq(chunk.sentence_2.map(normalize_text))).sum())
        chunk=chunk.assign(_priority=rng.random(len(chunk)))
        sample=pd.concat([sample,chunk],ignore_index=True).nsmallest(sample_size,'_priority')
    if not counts['rows']:raise ValueError(f'Empty CSV: {path}')
    # Exact quantiles from the integer word-count histogram, without retaining all rows.
    lengths=[]
    for col,hist in length_hist.items():
        values=np.array(sorted(hist));freq=np.array([hist[v] for v in values]);cum=freq.cumsum()
        row={'column':col,'mean_words':length_totals[col]/counts['rows'],'min_words':int(values[0]),'max_words':int(values[-1])}
        for q in (.5,.9,.95,.99):row[f'p{int(q*100)}_words']=int(values[np.searchsorted(cum,max(1,int(np.ceil(q*cum[-1]))))])
        row['rows_le_5_words']=sum(n for v,n in hist.items() if v<=5)
        row['rows_gt_128_words']=sum(n for v,n in hist.items() if v>128)
        lengths.append(row)
    report={'file':Path(path).name,**dict(counts),'missing':dict(missing),'blank_including_missing':dict(empty),
            'categories':{k:dict(v) for k,v in categories.items()},'word_lengths':lengths,
            'source_labels':[{'source':s,'label':int(l) if l in (0,1) else (str(l) if pd.notna(l) else None),'rows':int(n)} for (s,l),n in source_labels.items()]}
    return report,sample.drop(columns='_priority').reset_index(drop=True)


def text_features(series):
    text=series.fillna('').astype(str);chars=text.str.len().clip(lower=1)
    return pd.DataFrame({'words':text.str.split().str.len(),'characters':text.str.len(),
        'digit_ratio':text.str.count(r'\d')/chars,'punctuation_ratio':text.str.count(r'[^\w\s]')/chars,
        'uppercase_ratio':text.str.count(r'[A-Z]')/chars,
        'unique_word_ratio':text.map(lambda s:len(set(s.lower().split()))/max(1,len(s.split())))})


def lexical_pairs(frame, a, b, vectorizer):
    left=frame[a].fillna('').astype(str);right=frame[b].fillna('').astype(str)
    va=vectorizer.transform(left);vb=vectorizer.transform(right)
    cosine=np.asarray(va.multiply(vb).sum(axis=1)).ravel()
    def jaccard(x,y):
        x=set(x.lower().split());y=set(y.lower().split())
        return len(x&y)/max(1,len(x|y))
    return pd.DataFrame({'cosine':cosine,'word_jaccard':[jaccard(x,y) for x,y in zip(left,right)],
        'length_ratio':right.str.split().str.len()/left.str.split().str.len().clip(lower=1),
        'exact_equal':[normalize_text(x)==normalize_text(y) for x,y in zip(left,right)]},index=frame.index)


def disk_overlap_audit(specs, database_path):
    """Full normalized-text/group audit on disk. specs: (path, split, text_cols, group_cols)."""
    path=Path(database_path)
    if path.exists():raise FileExistsError('Choose a new audit database path')
    path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE texts (hash BLOB, split TEXT, PRIMARY KEY(hash,split)) WITHOUT ROWID')
        db.execute('CREATE TABLE groups_seen (id TEXT, split TEXT, PRIMARY KEY(id,split)) WITHOUT ROWID')
        db.execute('CREATE TABLE labels (hash BLOB, label INTEGER, PRIMARY KEY(hash,label)) WITHOUT ROWID')
        for file,split,text_cols,group_cols in specs:
            for chunk in pd.read_csv(file,chunksize=10000,dtype={c:str for c in group_cols}):
                for col in text_cols:
                    if chunk[col].isna().any() or chunk[col].astype(str).str.strip().eq('').any():raise ValueError(f'Missing/blank text in {file}:{col}')
                    hashes=[hashlib.sha256(normalize_text(s).encode()).digest() for s in chunk[col]]
                    db.executemany('INSERT OR IGNORE INTO texts VALUES (?,?)',((h,split) for h in hashes))
                    if text_cols==['sentence']:
                        if not chunk.label.isin([0,1]).all():raise ValueError(f'Invalid labels: {file}')
                        db.executemany('INSERT OR IGNORE INTO labels VALUES (?,?)',((h,int(l)) for h,l in zip(hashes,chunk.label)))
                for col in group_cols:
                    if chunk[col].isna().any():raise ValueError(f'Missing group: {file}')
                    db.executemany('INSERT OR IGNORE INTO groups_seen VALUES (?,?)',((g,split) for g in chunk[col]))
                db.commit()
            print('Audit scanned:',Path(file).name,flush=True)
        queries={'cross_split_texts':'SELECT COUNT(*) FROM (SELECT hash FROM texts GROUP BY hash HAVING COUNT(*)>1)',
                 'cross_split_groups':'SELECT COUNT(*) FROM (SELECT id FROM groups_seen GROUP BY id HAVING COUNT(*)>1)',
                 'conflicting_single_labels':'SELECT COUNT(*) FROM (SELECT hash FROM labels GROUP BY hash HAVING COUNT(*)>1)'}
        result={k:db.execute(q).fetchone()[0] for k,q in queries.items()}
        result['status']='passed' if not any(result.values()) else 'failed'
        return result
