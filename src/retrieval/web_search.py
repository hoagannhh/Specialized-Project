"""Search outcomes are explicit; errors must never masquerade as zero matches."""
import os
import requests
from src.utils import config
ENDPOINT='https://www.googleapis.com/customsearch/v1'


def search_web(query,max_results=config.WEB_SEARCH_MAX_RESULTS):
    if not isinstance(query,str) or not query.strip():
        return {'status':'error','items':[],'error':'empty_query'}
    if not 1<=max_results<=10:raise ValueError('max_results must be between 1 and 10')
    key,cx=os.getenv('GOOGLE_SEARCH_API_KEY'),os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    if not key or not cx:return {'status':'error','items':[],'error':'missing_credentials'}
    try:
        response=requests.get(ENDPOINT,params={'key':key,'cx':cx,'q':query,'num':max_results},
                              timeout=config.WEB_SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        data=response.json()
    except requests.RequestException as exc:
        # Never log exception URLs: they contain the API key and document text.
        return {'status':'error','items':[],'error':type(exc).__name__}
    except ValueError:
        return {'status':'error','items':[],'error':'invalid_json'}
    if not isinstance(data,dict) or 'error' in data or not isinstance(data.get('items',[]),list):
        return {'status':'error','items':[],'error':'invalid_api_response'}
    items=[];seen=set()
    for item in data.get('items',[]):
        if not isinstance(item,dict):continue
        url=item.get('link','')
        if not isinstance(url,str) or not url.startswith(('https://','http://')) or url in seen:continue
        seen.add(url)
        items.append({'url':url,'title':str(item.get('title','')),'snippet':str(item.get('snippet',''))})
    return {'status':'ok' if items else 'no_results','items':items,'error':None}
