"""Central runtime configuration; thresholds for trained classifiers live in checkpoints."""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR, PROCESSED_DIR, SPLITS_DIR = (DATA_DIR / n for n in ('raw','processed','splits'))
CROSS_EVAL_DIR, MODELS_DIR = DATA_DIR/'cross_eval', DATA_DIR/'models'
CHECKPOINTS_DIR = BASE_DIR/'checkpoints'
TANG0_CHECKPOINT_DIR, TANG3_CHECKPOINT_DIR, TANG4_CHECKPOINT_DIR = (CHECKPOINTS_DIR/f'tang{i}' for i in (0,3,4))
TANG1_VECTORIZER_PATH = MODELS_DIR/'tang1'/'tfidf_vectorizer.pkl'
FUSION_PATH = MODELS_DIR/'fusion.pkl'
TANG2_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TANG5_PROVIDER = os.getenv('TANG5_PROVIDER','openai')
TANG5_ENABLED_BY_DEFAULT = True
WEB_SEARCH_MAX_RESULTS = 3
WEB_SEARCH_TIMEOUT_SECONDS = 10
EVALUATION_DIR = BASE_DIR/'evaluation'
EDA_RESULTS_DIR = EVALUATION_DIR/'eda_results'

LLM_KEY_VARIABLES = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'gemini': 'GEMINI_API_KEY',
}


def configured_llm_keys(provider):
    """Đọc key theo thứ tự cấu hình, bỏ mục rỗng, key mẫu và key trùng."""
    if provider not in LLM_KEY_VARIABLES:
        return ()
    variables = [LLM_KEY_VARIABLES[provider]]
    if provider == 'gemini':
        variables.insert(0, 'GEMINI_API_KEYS')
    for variable in variables:
        value = os.getenv(variable, '')
        entries = value.split(',') if variable == 'GEMINI_API_KEYS' else [value]
        cleaned = (entry.strip() for entry in entries)
        keys = tuple(dict.fromkeys(key for key in cleaned
                                   if key and not re.fullmatch(r'(?:sk-(?:ant-)?)?x{4,}', key, re.I)))
        if keys:
            return keys
    return ()


def ensure_all_dirs():
    for path in (RAW_DIR,PROCESSED_DIR,SPLITS_DIR,CROSS_EVAL_DIR,MODELS_DIR,CHECKPOINTS_DIR,EDA_RESULTS_DIR):
        path.mkdir(parents=True,exist_ok=True)


def check_required_keys(require_tang5=False, require_web_search=False, provider=None, api_key=None):
    provider = provider or TANG5_PROVIDER
    if provider not in LLM_KEY_VARIABLES:
        raise ValueError('Unsupported LLM provider')
    required=[]
    if require_web_search:
        required += ['GOOGLE_SEARCH_API_KEY','GOOGLE_SEARCH_ENGINE_ID']
    missing=[k for k in required if not os.getenv(k)]
    if require_tang5 and not api_key and not configured_llm_keys(provider):
        missing.append('GEMINI_API_KEYS / GEMINI_API_KEY' if provider == 'gemini' else LLM_KEY_VARIABLES[provider])
    if missing:
        raise EnvironmentError('Missing configuration: '+', '.join(missing))
