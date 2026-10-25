"""Two retrieval modes; every candidate uses the same complete five-tier pipeline."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import Counter
from src.utils import config
from src.document_processing.file_reader import read_document
from src.document_processing.sentence_splitter import split_sentences_with_position
from src.retrieval.candidates import select_candidates

LABELS={'skipped_budget':'chưa kiểm tra nguồn (chế độ tiết kiệm)',
        'search_failed':'tìm kiếm bị lỗi','no_source':'chưa tìm thấy nguồn',
        'extraction_failed':'không đọc được nguồn','no_candidate':'chưa tìm thấy đoạn ứng viên',
        'retrieval_failed':'lỗi chọn đoạn ứng viên','model_failed':'lỗi phân tích cặp câu',
        'incomplete':'chưa kiểm tra đủ các nguồn ứng viên'}


class SingleFilePipeline:
    def __init__(self,mode='full',tang0_checkpoint_dir=config.TANG0_CHECKPOINT_DIR,
                 max_search_results=None,candidates_per_page=None,enable_tang5=True,
                 tang5_provider=None,fusion_path=config.FUSION_PATH,
                 classifier=None,search=None,fetch=None,cascade=None,semantic_scorer=None):
        if mode not in ('full','economy'):raise ValueError('mode must be full or economy')
        if not enable_tang5:raise ValueError('Both modes require all five tiers')
        self.mode=mode
        self.max_search_results=max_search_results if max_search_results is not None else (5 if mode=='full' else 3)
        self.top_k=candidates_per_page if candidates_per_page is not None else (3 if mode=='full' else 1)
        if not 1<=self.max_search_results<=10 or self.top_k<1:raise ValueError('Invalid retrieval budget')
        if classifier is None:
            from src.models.tang0_style_classifier.inference import predict_batch
            classifier=lambda sentences:predict_batch(sentences,tang0_checkpoint_dir).to_dict('records')
        if search is None:
            config.check_required_keys(require_web_search=True)
            from src.retrieval.web_search import search_web
            search=search_web
        if fetch is None:
            from src.retrieval.extract_page_text import fetch_and_extract_text
            fetch=fetch_and_extract_text
        if cascade is None:
            from src.pipeline.cascade_pipeline import CaTEXPipeline
            cascade=CaTEXPipeline(tang5_provider=tang5_provider,fusion_path=fusion_path)
        self.classifier,self.search,self.fetch,self.cascade=classifier,search,fetch,cascade
        self.semantic_scorer=semantic_scorer or cascade.semantic_scores

    def process_file(self,file_path):
        document=read_document(file_path)
        return self.process_document(document)

    def process_document(self,document):
        text=document['text'];spans=[]
        for block in document.get('paragraph_ranges') or [{'start':0,'end':len(text)}]:
            for span in split_sentences_with_position(text[block['start']:block['end']]):
                spans.append({**span,'start':span['start']+block['start'],'end':span['end']+block['start']})
        if not spans:raise ValueError('Không có câu đọc được')
        predictions=self.classifier([s['sentence'] for s in spans])
        if len(predictions)!=len(spans):raise ValueError('Wrong number of step-0 predictions')
        results=[];search_cache={};page_cache={};pair_cache={}
        for sentence_id,(span,prediction) in enumerate(zip(spans,predictions)):
            record={**span,'sentence_id':sentence_id,'tang0_probability':float(prediction['probability']),
                    'probability':None,'source_url':None,'status':None,'errors':[],'evidence':[]}
            if self.mode=='economy' and prediction['decision']=='skip':
                record['status']='skipped_budget';record['label']=LABELS[record['status']];results.append(record);continue
            query=span['sentence']
            if query not in search_cache:search_cache[query]=self.search(query,max_results=self.max_search_results)
            response=search_cache[query]
            if response['status']!='ok':
                record['status']='no_source' if response['status']=='no_results' else 'search_failed'
                record['errors']=[response.get('error')] if response.get('error') else []
            else:
                extracted=0;candidate_count=0;successful=[]
                for item in response['items']:
                    url=item['url']
                    if url not in page_cache:page_cache[url]=self.fetch(url)
                    page=page_cache[url]
                    if not page['success']:
                        record['errors'].append({'stage':'extract','url':url,'error':page.get('error')});continue
                    extracted+=1
                    try:candidates=select_candidates(query,page['text'],self.semantic_scorer,self.top_k)
                    except Exception as exc:
                        record['errors'].append({'stage':'retrieve','url':url,'error':type(exc).__name__});continue
                    candidate_count+=len(candidates)
                    for candidate in candidates:
                        key=query,candidate['sentence']
                        try:
                            if key not in pair_cache:pair_cache[key]=self.cascade.predict(*key)
                            pair=pair_cache[key]
                            successful.append({**pair,'source_url':url,'source_title':item.get('title',''),
                                'matched_source_sentence':candidate['sentence'],
                                'source_start':candidate['start'],'source_end':candidate['end']})
                        except Exception as exc:
                            record['errors'].append({'stage':'model','url':url,'error':type(exc).__name__})
                if successful:
                    successful.sort(key=lambda r:r['probability'],reverse=True)
                    best=successful[0]
                    record.update({k:v for k,v in best.items() if k not in ('status','errors')})
                    record['status']=best['decision']
                    if record['status']=='checked_negative' and record['errors']:record['status']='incomplete'
                    record['evidence']=successful
                    record['complete']=not record['errors'] and all(r.get('complete',False) for r in successful)
                elif not extracted:record['status']='extraction_failed'
                elif any(e['stage']=='retrieve' for e in record['errors']):record['status']='retrieval_failed'
                elif candidate_count:record['status']='model_failed'
                else:record['status']='no_candidate'
            record['label']=LABELS.get(record['status'],record.get('label','chưa đủ bằng chứng'))
            results.append(record)
        counts=dict(Counter(r['status'] for r in results))
        return {'file_path':document['file_path'],'format':document.get('format','txt'),
                'original_text':text,'paragraph_ranges':document.get('paragraph_ranges',[]),
                'warnings':document.get('warnings',[]),'mode':self.mode,
                'total_sentences':len(spans),'checked_sentences':sum(bool(r['evidence']) for r in results),
                'suspicious_sentences':counts.get('suspicious',0),'status_counts':counts,'results':results}


def main():
    import argparse,json
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='output/report.html')
    parser.add_argument('--mode',choices=['full','economy'],default='full')
    parser.add_argument('--fusion',default=str(config.FUSION_PATH))
    parser.add_argument('--provider',choices=['openai','anthropic'],default=config.TANG5_PROVIDER)
    args=parser.parse_args()
    output=Path(args.output)
    if output.suffix.lower() not in ('.html','.docx'):parser.error('--output must end in .html or .docx')
    if output.resolve()==Path(args.input).resolve():parser.error('Output must differ from input')
    result=SingleFilePipeline(mode=args.mode,tang5_provider=args.provider,fusion_path=args.fusion).process_file(args.input)
    from src.document_processing.highlight_exporter import export_highlighted_html,export_highlighted_docx
    output=Path(args.output)
    if output.suffix.lower()=='.docx':export_highlighted_docx(result,output)
    elif output.suffix.lower()=='.html':export_highlighted_html(result,output)
    else:parser.error('--output must end in .html or .docx')
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Report: {output.resolve()}')
    print(result['status_counts'])


if __name__=='__main__':main()
