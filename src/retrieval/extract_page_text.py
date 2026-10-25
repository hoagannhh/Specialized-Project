"""Extract HTML or text PDF sources with bounded downloads and explicit failures."""
from io import BytesIO
import requests
from bs4 import BeautifulSoup
from src.utils.config import WEB_SEARCH_TIMEOUT_SECONDS
MAX_BYTES=10*1024*1024


def extract_content(content,content_type):
    if 'pdf' in content_type:
        import pdfplumber
        with pdfplumber.open(BytesIO(content)) as pdf:
            pages=[page.extract_text() or '' for page in pdf.pages]
        if any(not text.strip() for text in pages):
            raise ValueError('pdf_requires_ocr_or_has_unreadable_pages')
        return '\n\n'.join(pages)
    if 'html' not in content_type:raise ValueError('unsupported_content_type')
    soup=BeautifulSoup(content,'html.parser')
    for element in soup(['script','style','nav','header','footer','aside','form','noscript']):element.decompose()
    root=soup.find('article') or soup.find('main') or soup.body or soup
    blocks=root.find_all(['p','li','h1','h2','h3','td','th'])
    texts=[b.get_text(' ',strip=True) for b in blocks if not b.find(['p','li','td','th'])]
    return '\n'.join(t for t in texts if t) if texts else root.get_text(' ',strip=True)


def fetch_and_extract_text(url,timeout=WEB_SEARCH_TIMEOUT_SECONDS):
    if not isinstance(url,str) or not url.startswith(('https://','http://')):
        return {'url':url,'text':'','success':False,'error':'invalid_url'}
    try:
        with requests.get(url,headers={'User-Agent':'CaTE-X Research Document Checker'},timeout=timeout,stream=True) as response:
            response.raise_for_status()
            chunks=[];size=0
            for chunk in response.iter_content(65536):
                size+=len(chunk)
                if size>MAX_BYTES:raise ValueError('source_too_large')
                chunks.append(chunk)
            text=extract_content(b''.join(chunks),response.headers.get('Content-Type','').lower())
        if not text.strip():raise ValueError('no_readable_content')
        return {'url':url,'text':text,'success':True,'error':None}
    except (requests.RequestException,ValueError) as exc:
        error=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        return {'url':url,'text':'','success':False,'error':error}
    except Exception as exc:
        return {'url':url,'text':'','success':False,'error':type(exc).__name__}


def fetch_and_extract_batch(urls,timeout=WEB_SEARCH_TIMEOUT_SECONDS):
    return [fetch_and_extract_text(url,timeout) for url in urls]
