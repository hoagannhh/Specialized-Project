import csv,hashlib,json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
def h(s): return hashlib.blake2b(' '.join(s.split()).encode(),digest_size=16).digest()
for name,cols in [('triplet_data',['anchor','positive','negative']),('pairwise_labeled',['sentence_1','sentence_2'])]:
    saved={}
    for split in ['train','val','test']:
        sets={c:set() for c in cols};n=0;labels={}
        with Path(f'data/splits/{name}_{split}.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                n+=1
                for c in cols: sets[c].add(h(r[c]))
                if 'label' in r: labels[r['label']]=labels.get(r['label'],0)+1
        print(name,split,'rows',n,'unique', {c:len(v) for c,v in sets.items()},'labels',labels,flush=True)
        if split=='train': saved=sets
        else:
            print('OVERLAP train/'+split,{a+'/'+b:len(saved[a]&sets[b]) for a in cols for b in cols},flush=True)
    del saved,sets
