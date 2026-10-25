"""Evaluate an exported document JSON against sentence_id,label CSV; no API calls."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import pandas as pd
from evaluation.common import binary_metrics,write_report


def evaluate_report(report,labels):
    rows=pd.DataFrame(report['results'])
    if rows.sentence_id.duplicated().any() or labels.sentence_id.duplicated().any():raise ValueError('Duplicate sentence IDs')
    if set(rows.sentence_id)!=set(labels.sentence_id):raise ValueError('Labels must cover exactly the report sentence IDs')
    if not set(labels.label).issubset({0,1}):raise ValueError('Labels must be binary')
    rows=rows.drop(columns=['label'],errors='ignore').merge(labels[['sentence_id','label']],on='sentence_id',validate='one_to_one')
    decided=rows.status.isin(['suspicious','checked_negative'])
    subset=rows.loc[decided]
    detected=(rows.status=='suspicious') & (rows.label==1)
    return {'total':len(rows),'decided':int(decided.sum()),'decision_coverage':float(decided.mean()),
        'unknown_or_skipped':int((~decided).sum()),'status_counts':rows.status.value_counts().to_dict(),
        'recall_over_all_labeled_positives':float(detected.sum()/(rows.label==1).sum()) if (rows.label==1).any() else None,
        'decided_subset_metrics':binary_metrics(subset.label,subset.probability,(subset.status=='suspicious').astype(int)) if len(subset) else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True);parser.add_argument('--labels',required=True)
    parser.add_argument('--output',default='evaluation/results/document.json');args=parser.parse_args()
    report=json.loads(Path(args.report).read_text(encoding='utf-8'))
    write_report(evaluate_report(report,pd.read_csv(args.labels)),args.output)


if __name__=='__main__':main()
