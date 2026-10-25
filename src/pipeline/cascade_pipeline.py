"""Every candidate runs tiers 1–5. LLM explains but never changes the decision."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from time import perf_counter
from src.utils import config
from src.pipeline.fusion import ScoreFusion, features, artifact_signature


class CaTEXPipeline:
    def __init__(self,tang1_vectorizer_path=config.TANG1_VECTORIZER_PATH,
                 tang2_model_name=config.TANG2_MODEL_NAME,
                 tang3_checkpoint_dir=config.TANG3_CHECKPOINT_DIR,
                 tang4_checkpoint_dir=config.TANG4_CHECKPOINT_DIR,
                 fusion_path=config.FUSION_PATH, enable_tang5=True,
                 tang5_provider=None,tang5_api_key=None,verbose=False,
                 scorers=None,judge=None,fusion=None):
        if not enable_tang5:
            raise ValueError('Production pipeline requires all five tiers. Use evaluation scripts for ablations.')
        self.provider=tang5_provider or config.TANG5_PROVIDER
        self.api_key=tang5_api_key
        self.paths=(tang1_vectorizer_path,tang3_checkpoint_dir,tang4_checkpoint_dir,tang2_model_name)
        self.model2_name=tang2_model_name
        if fusion is None:
            signature=artifact_signature(*self.paths)
            fusion=ScoreFusion.load(fusion_path,signature)
        self.fusion=fusion
        self.scorers=scorers
        self.judge=judge
        if judge is None:
            config.check_required_keys(require_tang5=True,provider=self.provider,api_key=self.api_key)
        if scorers is not None and len(scorers)!=4:
            raise ValueError('Exactly four scorers required')

    def _load_scorers(self):
        if self.scorers is not None:return
        from src.models.tang1_tfidf.filter import load_vectorizer,tfidf_filter
        from src.models.tang2_sbert_pretrained.encode import load_model,sbert_filter
        from src.models.tang3_triplet_encoder.inference import tang3_predict
        from src.models.tang4_cross_encoder.inference import tang4_predict
        vectorizer=load_vectorizer(self.paths[0])
        self.model2,self.tokenizer2=load_model(self.model2_name)
        self.scorers=[lambda a,b:tfidf_filter(a,b,vectorizer),
                      lambda a,b:sbert_filter(a,b,self.model2,self.tokenizer2),
                      lambda a,b:tang3_predict(a,b,self.paths[1]),
                      lambda a,b:tang4_predict(a,b,self.paths[2])]

    def semantic_scores(self,query,candidates):
        from src.models.tang2_sbert_pretrained.encode import load_model,sbert_filter_batch
        model,tokenizer=load_model(self.model2_name)
        return sbert_filter_batch([(query,c) for c in candidates],model,tokenizer).similarity.to_numpy()

    def predict(self,sentence_a,sentence_b):
        """Compatibility order: suspect first, source second."""
        if any(not isinstance(s,str) or not s.strip() for s in (sentence_a,sentence_b)):
            raise ValueError('Both sentences must be non-empty')
        self._load_scorers()
        trace=[];start=perf_counter();results=[]
        for tier,scorer in enumerate(self.scorers,1):
            # Training pairs are (source/anchor, suspect/candidate); cross-encoder is not symmetric.
            tick=perf_counter();result=scorer(sentence_b,sentence_a);results.append(result)
            trace.append({'tang':tier,'status':'ok','result':result,'time_ms':(perf_counter()-tick)*1000})
        values=features(results)
        final=self.fusion.predict(values)
        label='nghi vấn paraphrase' if final['decision']=='suspicious' else 'không phát hiện quan hệ paraphrase'
        tick=perf_counter()
        try:
            if self.judge is None:
                from src.models.tang5_llm_judge.call_api import tang5_judge
                judge=tang5_judge
            else:judge=self.judge
            explanation=judge(sentence_a,sentence_b,model_label=label,model_probability=final['probability'],
                              provider=self.provider,api_key=self.api_key)
            if not isinstance(explanation,dict) or explanation.get('parse_error') or not explanation.get('explanation'):
                raise ValueError('Invalid explanation')
            trace.append({'tang':5,'status':'ok','result':explanation,'time_ms':(perf_counter()-tick)*1000})
            explanation_status='ok'
        except Exception as exc:
            explanation={'explanation':None,'changed_phrases':[]}
            explanation_status='failed'
            trace.append({'tang':5,'status':'error','error':type(exc).__name__,'time_ms':(perf_counter()-tick)*1000})
        return {**final,'label':label,'scores':dict(zip(('lexical_similarity','pretrained_similarity','triplet_similarity','cross_logit'),values)),
                'explanation':explanation['explanation'],'changed_phrases':explanation.get('changed_phrases',[]),
                'explanation_status':explanation_status,'complete':explanation_status=='ok',
                'trace':trace,'total_time_ms':(perf_counter()-start)*1000}
