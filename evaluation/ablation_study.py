"""Offline diagnostic ablation of tier scores; does not change production's five tiers."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from src.pipeline.fusion import ScoreFusion
from evaluation.fit_fusion import validation_partition,score_pairs
from evaluation.common import binary_metrics,write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',required=True);parser.add_argument('--test',required=True)
    parser.add_argument('--output',default='evaluation/results/ablation.json')
    parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.batch_size<1 or (args.limit is not None and args.limit<1):parser.error('Sizes must be positive')
    frames=[pd.read_csv(p,nrows=args.limit,dtype={'group_id':str,'candidate_group_id':str}) for p in (args.validation,args.test)]
    groups=lambda f:set(f.group_id)|set(f.candidate_group_id)
    if groups(frames[0]) & groups(frames[1]):raise ValueError('Validation/test source groups overlap')
    fit,threshold=validation_partition(frames[0]);test=frames[1]
    if any(set(f.label)!={0,1} for f in (fit,threshold)):raise ValueError('Both labels required per validation partition')
    scores=[score_pairs(f,args.batch_size) for f in (fit,threshold,test)]
    results={}
    variants={'tier1_only':[0],'tier2_only':[1],'tier3_only':[2],'tier4_only':[3],
              'without_tier1':[1,2,3],'without_tier2':[0,2,3],'without_tier3':[0,1,3],'all_four':[0,1,2,3]}
    for name,columns in variants.items():
        masked=[]
        for values in scores:
            array=np.zeros_like(values);array[:,columns]=values[:,columns];masked.append(array)
        fusion=ScoreFusion.fit(masked[0],fit.label.to_numpy(),masked[1],threshold.label.to_numpy())
        p=fusion.model.predict_proba(masked[2])[:,1]
        results[name]={**binary_metrics(test.label,p,p>=fusion.threshold),'threshold':fusion.threshold}
    write_report({'note':'Offline comparison; tier 5 explains and cannot change labels. No LLM calls.',
                  'limit':args.limit,'variants':results},args.output)


if __name__=='__main__':main()
