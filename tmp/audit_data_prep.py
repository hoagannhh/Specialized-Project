import sys,csv,json,hashlib,unicodedata
from pathlib import Path
from collections import Counter,defaultdict
sys.stdout.reconfigure(encoding='utf-8')
def norm(s):return ' '.join(unicodedata.normalize('NFC',str(s)).split())
def key(s):return hashlib.blake2b(norm(s).encode(),digest_size=16).digest()
raw={}; sets={}; labels={}
for task in range(1,6):
 rows=json.loads(Path(f'data/raw/padben_sentence_pair_task{task}.json').read_text(encoding='utf-8-sig'))
 raw[task]=rows;sets[task]=set();labels[task]=defaultdict(set)
 for r in rows:
  for s,l in zip(r['sentence_pair'],r['label_pair']):
   k=key(s);sets[task].add(k);labels[task][k].add(l)
 print('RAW',task,'rows',len(rows),'schema',list(rows[0]),'labels',dict(Counter(l for r in rows for l in r['label_pair'])),flush=True)
for a,b in [(1,3),(2,5),(3,4),(4,5),(1,2),(1,4),(3,5)]:
 inter=sets[a]&sets[b]
 conflict=sum(bool(labels[a][k] != labels[b][k]) for k in inter)
 print('TASK_OVERLAP',a,b,len(inter),'different_raw_labels',conflict,flush=True)
allpad=set.union(*sets.values()); splitkeys={}; groups={}; padseen={}
for split in ['train','val','test']:
 ks=set();gs=set();ps=set();counts=Counter();plabels=defaultdict(set);dup=0;task3zero=0;n=0
 with open(f'data/splits/tang0_{split}.csv',encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   n+=1;k=key(r['sentence']);dup+=k in ks;ks.add(k);gs.add((r['source'],r['group_id']));counts[(r['source'],r['label'])]+=1
   if k in allpad:ps.add(k);plabels[k].add(r['label'])
   if r['source']=='padben_task3' and r['label']=='0':task3zero+=1
 splitkeys[split]=ks;groups[split]=gs;padseen[split]=ps
 print('SPLIT',split,'rows',n,'unique_text',len(ks),'duplicate_rows',dup,'counts',dict(counts),'conflicting_pad_text',sum(len(v)>1 for v in plabels.values()),'task3zero',task3zero,flush=True)
for a,b in [('train','val'),('train','test'),('val','test')]:
 print('SPLIT_OVERLAP',a,b,'texts',len(splitkeys[a]&splitkeys[b]),'groups',len(groups[a]&groups[b]),'pad_texts',len(padseen[a]&padseen[b]),flush=True)
for task in [4,5]:
 print('HELDOUT_OVERLAP',task,'unique',len(sets[task]),'in_train',len(sets[task]&splitkeys['train']),'in_val',len(sets[task]&splitkeys['val']),flush=True)
for task in [3]:
 with open(f'data/processed/padben/single_task{task}.csv',encoding='utf-8-sig',newline='') as f:
  rows=list(csv.DictReader(f))
 print('PROCESSED_TASK',task,'labels',dict(Counter(r['label'] for r in rows)),flush=True)
