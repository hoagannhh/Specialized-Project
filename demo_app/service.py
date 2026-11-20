"""Dịch vụ demo: khảo sát tầng 3 độc lập và kết nối pipeline đầy đủ."""
import base64
import json
import os
from functools import lru_cache
from pathlib import Path
import zipfile

from src.utils import config
from src.document_processing.file_reader import read_document
from src.document_processing.sentence_splitter import split_sentences_with_position

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output' / 'demo'


@lru_cache(maxsize=1)
def _pair_pipeline():
    """Giữ pipeline cặp câu trong RAM giữa các lượt kiểm tra của demo."""
    from src.pipeline.cascade_pipeline import CaTEXPipeline
    return CaTEXPipeline()


@lru_cache(maxsize=2)
def _document_pipeline(retrieval):
    """Tái sử dụng pipeline tài liệu theo chế độ truy hồi."""
    from src.pipeline.single_file_pipeline import SingleFilePipeline
    return SingleFilePipeline(mode=retrieval)


def bundle_status(tier):
    directory = config.CHECKPOINTS_DIR / f'tang{tier}'
    names = [f'tang{tier}_best.pt', 'model_config.json', 'tokenizer_config.json']
    missing = [n for n in names if not (directory/n).is_file() or not (directory/n).stat().st_size]
    if not any((directory/n).is_file() and (directory/n).stat().st_size for n in ['tokenizer.json','vocab.txt','sentencepiece.bpe.model','spiece.model']):
        missing.append('tokenizer vocabulary')
    return {'ready': not missing, 'detail': 'Đã có bộ checkpoint' if not missing else 'Thiếu: '+', '.join(missing)}


def capabilities():
    tiers = {str(i): bundle_status(i) for i in (0,3,4)}
    tiers['1'] = {'ready': config.TANG1_VECTORIZER_PATH.is_file(),
                  'detail': 'Đã có vectorizer TF-IDF' if config.TANG1_VECTORIZER_PATH.is_file()
                  else f'Thiếu {config.TANG1_VECTORIZER_PATH.relative_to(ROOT)}'}
    tiers['2'] = {'ready': True,
                  'detail': f'Pretrained: {config.TANG2_MODEL_NAME}; lần đầu có thể cần tải model'}
    provider = config.TANG5_PROVIDER
    provider_keys = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY',
                     'gemini': 'GEMINI_API_KEY'}
    key_name = provider_keys.get(provider)
    key = os.getenv(key_name, '') if key_name else ''
    tiers['5'] = {'ready': bool(key) and 'xxxx' not in key,
                  'detail': (f'LLM: {provider} (đã cấu hình; chưa kiểm tra kết nối)' if
                             bool(key) and 'xxxx' not in key
                             else f'Chưa cấu hình API key cho LLM ({provider})')}
    fusion = config.FUSION_PATH.is_file()
    web = all(os.getenv(k) and 'xxxx' not in os.getenv(k) for k in ['GOOGLE_SEARCH_API_KEY','GOOGLE_SEARCH_ENGINE_ID'])
    fusion_detail = 'Đã có bộ kết hợp điểm' if fusion else f'Thiếu {config.FUSION_PATH.relative_to(ROOT)}'
    pair_missing = [tiers[str(i)]['detail'] for i in range(1,6) if not tiers[str(i)]['ready']]
    if not fusion: pair_missing.append(fusion_detail)
    document_missing = list(pair_missing)
    if not tiers['0']['ready']: document_missing.append('Bước 0: '+tiers['0']['detail'])
    if not web: document_missing.append('Chưa cấu hình Google Search API key và Search Engine ID')
    pair = not pair_missing
    return {'tiers': tiers, 'fusion': fusion, 'fusion_detail': fusion_detail, 'web': web,
            'full_pair': pair,
            'full_pair_missing': pair_missing,
            'full_document': not document_missing, 'full_document_missing': document_missing,
            'notice': 'Trạng thái dựa trên file/cấu hình; tính tương thích và kết nối được kiểm tra khi phân tích.'}


def document_input(value, directory, stem):
    if not isinstance(value, dict): raise ValueError('Đầu vào tài liệu không hợp lệ')
    upload = value.get('file')
    if upload:
        suffix = Path(str(upload.get('name',''))).suffix.lower()
        if suffix not in ('.txt','.docx','.pdf'): raise ValueError('Chỉ hỗ trợ TXT, DOCX, PDF')
        try: data = base64.b64decode(upload.get('data',''), validate=True)
        except Exception: raise ValueError('File tải lên không hợp lệ')
        if not data or len(data)>8*1024*1024: raise ValueError('Mỗi file phải từ 1 byte đến 8 MB')
        path = directory / (stem+suffix)
        path.write_bytes(data)
        if suffix == '.docx':
            with zipfile.ZipFile(path) as archive:
                if sum(i.file_size for i in archive.infolist()) > 50*1024*1024:
                    raise ValueError('DOCX có nội dung giải nén quá lớn cho bản demo')
        doc = read_document(path)
    else:
        text = value.get('text','')
        if not isinstance(text,str) or not text.strip(): raise ValueError('Hãy nhập văn bản hoặc chọn file')
        path=directory/(stem+'.txt');path.write_text(text,encoding='utf8');doc=read_document(path)
    if len(doc['text'])>40000: raise ValueError('Bản demo giới hạn 40.000 ký tự cho mỗi tài liệu')
    return doc


def spans_for(doc):
    spans=[]
    for block in doc.get('paragraph_ranges') or [{'start':0,'end':len(doc['text'])}]:
        for row in split_sentences_with_position(doc['text'][block['start']:block['end']]):
            spans.append({**row,'start':row['start']+block['start'],'end':row['end']+block['start']})
    return spans


def analyze(payload, job_id, progress):
    directory=OUTPUT/job_id;directory.mkdir(parents=True,exist_ok=True)
    mode=payload.get('mode','normal');kind=payload.get('kind','pair')
    if mode not in ('normal','deep') or kind not in ('pair','document'):raise ValueError('Chế độ không hợp lệ')
    cap=capabilities()
    readiness='full_pair' if kind=='pair' else 'full_document'
    if not cap[readiness]:
        missing=cap[readiness+'_missing']
        raise ValueError('Chưa đủ thành phần để chạy đủ 5 tầng: '+'; '.join(missing))
    if kind=='pair':
        suspect=payload.get('suspect','');source=payload.get('source','')
        if any(not isinstance(x,str) or not x.strip() or len(x)>8000 for x in [suspect,source]):raise ValueError('Nhập hai đoạn văn bản, mỗi đoạn tối đa 8.000 ký tự')
        pipeline_loaded = _pair_pipeline.cache_info().currsize > 0
        progress('Đang đối chiếu bằng pipeline đã nạp…' if pipeline_loaded
                 else 'Lượt đầu: đang khởi tạo pipeline và nạp model…')
        result={**_pair_pipeline().predict(suspect,source),'kind':kind,'mode':mode,
                'pipeline':'five_tier','source':source,'suspect':suspect,
                'warnings':['Cặp câu chạy đủ 5 tầng. Kết quả là gợi ý đối chiếu; điểm fusion chưa được hiệu chỉnh.']}
    else:
        progress('Đang đọc tài liệu…');doc=document_input(payload.get('document'),directory,'input')
        if len(spans_for(doc))>200:raise ValueError('Demo giới hạn 200 câu kiểm tra')
        from src.document_processing.highlight_exporter import export_highlighted_html,export_highlighted_docx
        retrieval='economy' if mode=='normal' else 'full'
        if mode=='normal':
            progress('Kiểm tra thường: Bước 0 chọn câu cần tìm nguồn; mỗi cặp chạy đủ 5 tầng…')
        else:
            progress('Kiểm tra sâu: tìm nguồn cho mọi câu; mỗi cặp chạy đủ 5 tầng…')
        result=_document_pipeline(retrieval).process_document(doc)
        export_highlighted_html(result,directory/'report.html');export_highlighted_docx(result,directory/'report.docx')
        result.update(kind=kind,mode=mode,retrieval=retrieval,pipeline='five_tier')
        result.setdefault('warnings',[]).append('Mỗi cặp nguồn được chấm qua đủ 5 tầng. Điểm fusion chưa được hiệu chỉnh; kết quả là gợi ý đối chiếu.')
    (directory/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    result['downloads']={suffix:f'/api/jobs/{job_id}/report.{suffix}' for suffix in ['json','html','docx'] if (directory/f'report.{suffix}').exists()}
    return result
