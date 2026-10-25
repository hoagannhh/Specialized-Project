"""Fit four-tier fusion on validation pairs; never pass the test split here."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from src.utils import config
from src.pipeline.fusion import ScoreFusion, FEATURES, artifact_signature


def validation_partition(frame, seed=42):
    """Keep both endpoints of a pair on the same side of the fit/threshold split."""
    required={'group_id','candidate_group_id','label'}
    if not required.issubset(frame.columns):
        raise ValueError('Validation CSV needs group_id, candidate_group_id and label')
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing group IDs or labels')
    def side(value):
        digest=hashlib.sha256(f'{seed}:{value}'.encode()).digest()
        return digest[0] < 128
    a=frame.group_id.map(side);b=frame.candidate_group_id.map(side)
    return frame.loc[a & b],frame.loc[~a & ~b]


def score_pairs(frame, batch_size=32):
    from src.models.tang1_tfidf.filter import load_vectorizer,tfidf_filter_batch
    from src.models.tang2_sbert_pretrained.encode import load_model,sbert_filter_batch
    from src.models.tang3_triplet_encoder.inference import tang3_predict_batch
    from src.models.tang4_cross_encoder.inference import tang4_predict_batch
    pairs=list(zip(frame.sentence_1,frame.sentence_2))
    vectorizer=load_vectorizer(config.TANG1_VECTORIZER_PATH)
    model,tokenizer=load_model(config.TANG2_MODEL_NAME)
    s1=tfidf_filter_batch(pairs,vectorizer).similarity.to_numpy()
    s2=sbert_filter_batch(pairs,model,tokenizer,batch_size=batch_size).similarity.to_numpy()
    s3=tang3_predict_batch(pairs,config.TANG3_CHECKPOINT_DIR,batch_size=batch_size).similarity.to_numpy()
    p=tang4_predict_batch(pairs,config.TANG4_CHECKPOINT_DIR,batch_size=batch_size).probability.to_numpy()
    p=np.clip(p,1e-7,1-1e-7)
    return np.column_stack([s1,s2,s3,np.log(p/(1-p))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',required=True,help='Pairwise validation CSV, not train/test')
    parser.add_argument('--output',default=str(config.FUSION_PATH))
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--limit',type=int,help='Optional smoke-test row limit; omit for real fitting')
    args=parser.parse_args()
    if args.batch_size<1 or (args.limit is not None and args.limit<1):parser.error('Sizes must be positive')
    frame=pd.read_csv(args.validation,nrows=args.limit,dtype={'group_id':str,'candidate_group_id':str})
    if not {'sentence_1','sentence_2'}.issubset(frame):raise ValueError('Expected sentence_1 and sentence_2')
    # Check partitions before expensive inference.
    fit,threshold=validation_partition(frame)
    if any(set(part.label)!={0,1} for part in (fit,threshold)):
        raise ValueError('Each partition must contain both classes; use more validation groups')
    signature=artifact_signature(config.TANG1_VECTORIZER_PATH,config.TANG3_CHECKPOINT_DIR,config.TANG4_CHECKPOINT_DIR,config.TANG2_MODEL_NAME)
    scored=[]
    for part in (fit,threshold):
        blocks=[]
        for start in range(0,len(part),1024):
            blocks.append(score_pairs(part.iloc[start:start+1024],args.batch_size))
            print(f'Scored {min(start+1024,len(part))}/{len(part)}',flush=True)
        scored.append(np.concatenate(blocks))
    fusion=ScoreFusion.fit(scored[0],fit.label.to_numpy(),scored[1],threshold.label.to_numpy(),signature)
    fusion.save(args.output)
    report={'validation':str(Path(args.validation).resolve()),'rows':len(frame),'fit_rows':len(fit),
            'threshold_rows':len(threshold),'discarded_cross_partition_pairs':len(frame)-len(fit)-len(threshold),
            'threshold':fusion.threshold,'features':FEATURES,'signature':signature,'probability_calibrated':False,
            'smoke_test_limit':args.limit}
    Path(args.output).with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
