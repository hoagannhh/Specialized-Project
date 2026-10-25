"""Evaluate ranking of anchor-positive versus anchor-negative on held-out triplets."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from evaluation.common import write_report
from src.models.tang3_triplet_encoder.inference import tang3_predict_batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--checkpoint',default='checkpoints/tang3')
    parser.add_argument('--output',default='evaluation/results/tang3.json');parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.batch_size<1 or (args.limit is not None and args.limit<1):parser.error('Sizes must be positive')
    frame=pd.read_csv(args.input,nrows=args.limit)
    if frame.empty:raise ValueError('Empty test data')
    positive=tang3_predict_batch(list(zip(frame.anchor,frame.positive)),args.checkpoint,args.batch_size).similarity
    negative=tang3_predict_batch(list(zip(frame.anchor,frame.negative)),args.checkpoint,args.batch_size).similarity
    write_report({'count':len(frame),'ranking_accuracy':float((positive>negative).mean()),'ties':int((positive==negative).sum()),
        'mean_positive_similarity':float(positive.mean()),'mean_negative_similarity':float(negative.mean()),
        'mean_gap':float((positive-negative).mean())},args.output)


if __name__=='__main__':main()
