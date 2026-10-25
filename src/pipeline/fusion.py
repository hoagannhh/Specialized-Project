"""Four-score logistic fusion. Fit only on validation/OOF scores, never test."""
import hashlib
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
FEATURES = ['lexical_similarity','pretrained_similarity','triplet_similarity','cross_logit']


def features(results):
    p=float(results[3]['probability'])
    if not 0 <= p <= 1:
        raise ValueError('Invalid cross-encoder probability')
    p=np.clip(p,1e-7,1-1e-7)
    values=[float(results[i]['similarity']) for i in range(3)]+[float(np.log(p/(1-p)))]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite tier scores')
    return values


def artifact_signature(vectorizer, checkpoint3, checkpoint4, model2):
    result={'tier2':model2}
    for name,path in [('tier1',Path(vectorizer)),('tier3',Path(checkpoint3)/'tang3_best.pt'),('tier4',Path(checkpoint4)/'tang4_best.pt')]:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        result[name]=digest.hexdigest()
    return result


class ScoreFusion:
    def __init__(self, model, threshold, signature=None):
        self.model, self.threshold, self.signature = model, float(threshold), signature

    @classmethod
    def fit(cls, x_fit, y_fit, x_threshold, y_threshold, signature=None):
        for x,y in [(x_fit,y_fit),(x_threshold,y_threshold)]:
            if np.asarray(x).ndim!=2 or np.asarray(x).shape[1]!=4 or not np.isfinite(x).all():
                raise ValueError('Expected finite four-column scores')
            if set(np.asarray(y).tolist())!={0,1}:raise ValueError('Both labels required')
        model=LogisticRegression(max_iter=2000,random_state=42).fit(x_fit,y_fit)
        probs=model.predict_proba(x_threshold)[:,1]
        precision,recall,thresholds=precision_recall_curve(y_threshold,probs)
        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        return cls(model,thresholds[int(np.argmax(f1))],signature)

    def predict(self, values):
        probability=float(self.model.predict_proba(np.asarray(values).reshape(1,4))[0,1])
        return {'probability':probability,'threshold_used':self.threshold,
                'decision':'suspicious' if probability>=self.threshold else 'checked_negative',
                'probability_calibrated':False}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:pickle.dump({'version':1,'features':FEATURES,'fusion':self},f)

    @classmethod
    def load(cls,path,signature=None):
        with Path(path).open('rb') as f:payload=pickle.load(f)
        if payload.get('version')!=1 or payload.get('features')!=FEATURES:
            raise ValueError('Unsupported fusion artifact')
        obj=payload['fusion']
        if signature is not None and obj.signature!=signature:
            raise ValueError('Model artifacts changed: refit fusion on validation scores')
        return obj
