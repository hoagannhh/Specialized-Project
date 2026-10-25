"""Hybrid page-level candidates: lexical and semantic ranks, plus two-sentence windows."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.document_processing.sentence_splitter import split_sentences_with_position


def select_candidates(query,page_text,semantic_scorer,top_k=3):
    if top_k<1:raise ValueError('top_k must be positive')
    spans=split_sentences_with_position(page_text,min_length=1)
    windows=[];seen=set()
    for i,span in enumerate(spans):
        for width in (1,2):
            if i+width>len(spans):continue
            end=spans[i+width-1]['end'];text=' '.join(page_text[span['start']:end].split())
            if not text or text in seen:continue
            seen.add(text);windows.append({'sentence':text,'start':span['start'],'end':end})
    if not windows:return []
    texts=[w['sentence'] for w in windows]
    try:
        matrix=TfidfVectorizer().fit_transform([query]+texts)
        lexical=(matrix[1:]@matrix[0].T).toarray().ravel()
    except ValueError:lexical=np.zeros(len(texts))
    semantic=np.asarray(semantic_scorer(query,texts),dtype=float)
    if semantic.shape!=(len(texts),) or not np.isfinite(semantic).all():
        raise ValueError('Invalid semantic retrieval scores')
    # Union retains semantic-only matches; zero lexical overlap is not an exclusion.
    rank={}
    for values in (lexical,semantic):
        order=np.argsort(-values,kind='stable')
        valid=[int(i) for i in order if values[i]>0][:top_k]
        for position,index in enumerate(valid):rank[index]=rank.get(index,0)+1/(60+position+1)
    return [{**windows[i],'lexical_score':float(lexical[i]),'semantic_score':float(semantic[i])}
            for i in sorted(rank,key=lambda i:-rank[i])]
