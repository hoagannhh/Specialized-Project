"""Offline binary evaluation with fixed model thresholds; no test-set tuning."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,average_precision_score,roc_auc_score


def binary_metrics(labels, probabilities, predictions):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float);pred=np.asarray(predictions,dtype=int)
    if not len(y) or len(y)!=len(p) or len(y)!=len(pred):raise ValueError('Non-empty aligned predictions required')
    if not set(y).issubset({0,1}) or not set(pred).issubset({0,1}) or not np.isfinite(p).all():raise ValueError('Invalid labels or scores')
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'count':len(y),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision),'recall':float(recall),
        'f1':float(f1),'confusion_matrix_labels_0_1':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'average_precision':float(average_precision_score(y,p)) if len(set(y))==2 else None,
        'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'single_class':len(set(y))<2}


def write_report(report,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def evaluate_classifier(tier):
    import argparse,pandas as pd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--checkpoint',default=f'checkpoints/tang{tier}')
    parser.add_argument('--output',default=f'evaluation/results/tang{tier}.json')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.batch_size<1 or (args.limit is not None and args.limit<1):parser.error('Sizes must be positive')
    frame=pd.read_csv(args.input,nrows=args.limit)
    if tier==0:
        from src.models.tang0_style_classifier.inference import predict_batch
        output=predict_batch(frame.sentence.tolist(),args.checkpoint,batch_size=args.batch_size)
    else:
        from src.models.tang4_cross_encoder.inference import tang4_predict_batch
        output=tang4_predict_batch(list(zip(frame.sentence_1,frame.sentence_2)),args.checkpoint,batch_size=args.batch_size)
    pred=(output.probability>=output.threshold_used).astype(int)
    report=binary_metrics(frame.label,output.probability,pred)
    report.update(input=str(Path(args.input).resolve()),threshold=float(output.threshold_used.iloc[0]),tier=tier)
    if 'original_task_label' in frame:
        report['by_original_task_label']={str(group):binary_metrics(frame.loc[indices,'label'],output.loc[indices,'probability'],pred.loc[indices]) for group,indices in frame.groupby('original_task_label').groups.items()}
    write_report(report,args.output)
