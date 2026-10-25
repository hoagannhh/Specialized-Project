"""Central runtime configuration; thresholds for trained classifiers live in checkpoints."""
import os
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


def ensure_all_dirs():
    for path in (RAW_DIR,PROCESSED_DIR,SPLITS_DIR,CROSS_EVAL_DIR,MODELS_DIR,CHECKPOINTS_DIR,EDA_RESULTS_DIR):
        path.mkdir(parents=True,exist_ok=True)


def check_required_keys(require_tang5=False, require_web_search=False, provider=None, api_key=None):
    provider = provider or TANG5_PROVIDER
    if provider not in ('openai','anthropic'):
        raise ValueError('Unsupported LLM provider')
    required=[]
    if require_web_search:
        required += ['GOOGLE_SEARCH_API_KEY','GOOGLE_SEARCH_ENGINE_ID']
    if require_tang5 and not api_key:
        required += ['OPENAI_API_KEY' if provider=='openai' else 'ANTHROPIC_API_KEY']
    missing=[k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError('Missing configuration: '+', '.join(missing))
