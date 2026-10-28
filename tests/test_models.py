"""Offline regression tests: real tiny Transformers, no downloads or paid APIs."""
import importlib
import inspect
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from src.models import runtime
from src.models.tang0_style_classifier.model import SingleSentenceClassifier
from src.models.tang3_triplet_encoder.model import BiEncoder, TripletLoss, DEFAULT_PRETRAINED_NAME
from src.models.tang4_cross_encoder.model import CrossEncoderModel
from src.models.tang0_style_classifier import inference as i0
from src.models.tang2_sbert_pretrained import encode as i2
from src.models.tang3_triplet_encoder import inference as i3
from src.models.tang4_cross_encoder import inference as i4
from src.models.tang1_tfidf import filter as i1
from src.models.tang5_llm_judge import call_api as i5


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(__file__).resolve().parents[1] / 'tmp' / ('models-test-' + uuid.uuid4().hex)
        cls.base = cls.root / 'tiny-bert'
        cls.base.mkdir(parents=True)
        tokens = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','cat','dog','runs','walks','the','a','bird','flies','.']
        vocab = {t:i for i,t in enumerate(tokens)}
        (cls.base/'vocab.txt').write_text('\n'.join(tokens),encoding='utf-8')
        cls.tokenizer = (BertTokenizer(vocab=vocab) if 'vocab' in inspect.signature(BertTokenizer).parameters
                         else BertTokenizer(vocab_file=str(cls.base/'vocab.txt')))
        cls.tokenizer.save_pretrained(cls.base)
        cls.config = BertConfig(vocab_size=len(tokens),hidden_size=16,num_hidden_layers=1,
                                num_attention_heads=2,intermediate_size=32,max_position_embeddings=64)
        BertModel(cls.config).save_pretrained(cls.base)
        cls.models = {}
        for tier,typ in [(0,SingleSentenceClassifier),(3,BiEncoder),(4,CrossEncoderModel)]:
            model=typ(encoder_config=cls.config.to_dict()).eval()
            directory=cls.root/f'tier{tier}'
            runtime.save_bundle(model,cls.tokenizer,directory,tier,str(cls.base),17,
                                threshold=None if tier==3 else 0.7)
            cls.models[tier]=(model,directory)

    def test_triplet_loss_and_masked_pooling(self):
        a=torch.tensor([[1.,0.]])
        b=torch.tensor([[0.,1.]])
        self.assertEqual(TripletLoss(.5)(a,a,b).item(),0.)
        self.assertEqual(TripletLoss(.5)(a,b,a).item(),1.5)
        pooled=self.models[3][0]._mean_pooling(torch.tensor([[[1.,3.],[99.,99.]]]),torch.tensor([[1,0]]))
        torch.testing.assert_close(pooled,torch.tensor([[1.,3.]]))
        self.assertEqual(DEFAULT_PRETRAINED_NAME,i2.DEFAULT_MODEL_NAME)

    def test_checkpoint_load_is_offline_and_exact(self):
        for tier,typ in [(0,SingleSentenceClassifier),(3,BiEncoder),(4,CrossEncoderModel)]:
            original,directory=self.models[tier]
            with patch('src.models.runtime.AutoModel.from_pretrained',side_effect=AssertionError('network initialization')):
                loaded,tokenizer,meta=runtime.load_bundle(directory,tier,typ,'cpu')
            self.assertEqual(meta['max_length'],17)
            for key,value in original.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key],value)
            self.assertEqual(tokenizer('the cat')['input_ids'],self.tokenizer('the cat')['input_ids'])

    def test_cache_respects_directory_and_overrides(self):
        model,directory=self.models[4]
        a,_=i4.load_cached_model(directory,device='cpu')
        other=self.root/'other-tier4'
        runtime.save_bundle(model,self.tokenizer,other,4,str(self.base),23,threshold=.2)
        b,_=i4.load_cached_model(other,device='cpu')
        self.assertIsNot(a,b)
        self.assertEqual(b._inference_metadata['max_length'],23)
        with self.assertRaises(ValueError):
            i4.load_cached_model(other,pretrained_name='wrong-backbone',device='cpu')

    def test_batch_equivalence_saved_length_and_no_early_exit(self):
        pairs=[('the cat runs','a cat walks'),('the dog walks','a bird flies'),('the cat','the cat')]
        model,directory=self.models[3]
        one=i3.tang3_predict(*pairs[0],checkpoint_dir=directory)
        many=i3.tang3_predict_batch(pairs,directory,batch_size=2)
        self.assertAlmostEqual(one['similarity'],many.similarity.iloc[0],places=5)
        self.assertTrue(many.needs_tang4.all())
        self.assertEqual(i2.sbert_filter(*pairs[2],model=model,tokenizer=self.tokenizer)['decision'],'continue')
        for module,tier,function in [(i0,0,'predict_batch'),(i4,4,'tang4_predict_batch')]:
            original,directory=self.models[tier]
            calls=[]
            tokenizer=self.tokenizer
            class TrackingTokenizer:
                def __call__(self,*args,**kwargs):
                    calls.append((len(args[0]),kwargs['max_length']))
                    return tokenizer(*args,**kwargs)
            original._inference_metadata={'max_length':17,'threshold':.7}
            loader='load_for_inference' if tier==0 else 'load_cached_model'
            result=(original,TrackingTokenizer(),.7) if tier==0 else (original,TrackingTokenizer())
            with patch.object(module,loader,return_value=result):
                values=[p[0] for p in pairs] if tier==0 else pairs
                out=getattr(module,function)(values,directory,batch_size=2)
            self.assertEqual(calls,[(2,17),(1,17)])
            self.assertTrue((out.threshold_used==.7).all())
        for function in [i2.sbert_filter_batch,i3.tang3_predict_batch,i4.tang4_predict_batch]:
            self.assertTrue(function([]).empty)
            with self.assertRaises(ValueError):function([('','valid')])

    def test_tfidf_cache_and_zero_scores_continue(self):
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer
        a=TfidfVectorizer().fit(['cat runs','dog walks'])
        b=TfidfVectorizer().fit(['bird flies','fish swims'])
        for name,value in [('a',a),('b',b)]:
            with (self.root/f'{name}.pkl').open('wb') as f:pickle.dump(value,f)
        self.assertNotEqual(i1.load_vectorizer(self.root/'a.pkl').vocabulary_, i1.load_vectorizer(self.root/'b.pkl').vocabulary_)
        result=i1.tfidf_filter('cat','dog',a)
        self.assertEqual(result['similarity'],0)
        self.assertEqual(result['decision'],'continue')

    def test_llm_schema_direction_and_evidence(self):
        for raw in ['[]','null','{"explanation":42,"changed_phrases":[]}']:
            self.assertTrue(i5._extract_json_from_text(raw)['parse_error'])
        valid=json.dumps({'changed_phrases':[{'original':'cat','changed_to':'dog'}],
                          'explanation':'Two different animals.', 'confidence':1, 'is_paraphrase':True})
        with patch.object(i5,'call_openai',return_value=valid) as api:
            result=i5.tang5_judge('the dog walks','the cat runs','nghi vấn paraphrase',.7)
        prompt=api.call_args.args[0]
        self.assertIn('"source_sentence": "the cat runs"',prompt)
        self.assertIn('"suspect_sentence": "the dog walks"',prompt)
        self.assertNotIn('confidence',result)
        self.assertFalse(result['parse_error'])
        with patch.object(i5,'call_openai',return_value=valid):
            self.assertTrue(i5.tang5_judge('bird flies','fish swims','negative',.2)['parse_error'])

    def test_tiny_training_all_three_tiers(self):
        frames={0:pd.DataFrame({'sentence':['the cat runs','a dog walks','the bird flies','a cat walks'],'label':[0,1,0,1]}),
                3:pd.DataFrame({'anchor':['the cat runs','the dog walks'], 'positive':['a cat walks','a dog runs'], 'negative':['the bird flies','a bird flies']}),
                4:pd.DataFrame({'sentence_1':['the cat runs','a dog walks','the bird flies','a cat walks'],
                                'sentence_2':['a cat walks','the bird flies','a bird flies','a dog walks'],'label':[1,0,1,0]})}
        for tier,folder in [(0,'tang0_style_classifier'),(3,'tang3_triplet_encoder'),(4,'tang4_cross_encoder')]:
            frame=frames[tier]
            csv=self.root/f'train{tier}.csv';frame.to_csv(csv,index=False)
            module=importlib.import_module(f'src.models.{folder}.train')
            output=self.root/f'trained{tier}'
            # Train and validation reuse tiny fixtures only to exercise code, not measure accuracy.
            result=module.train(str(csv),str(csv),str(output),pretrained_name=str(self.base),
                                max_length=17,batch_size=2,num_epochs=2,seed=7)
            model=result[0] if tier==0 else result
            payload=runtime.read_bundle(output,tier)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(),payload['state_dict'][key].cpu())
            self.assertEqual(payload['metadata']['seed'],7)
            self.assertTrue((output/'tokenizer_config.json').exists())


    def test_resume_mid_epoch_matches_uninterrupted_all_tiers(self):
        from src.models.training_checkpoint import TrainingRun
        frames = {
            0: pd.DataFrame({'sentence':['the cat','a dog','the bird','a cat'], 'label':[0,1,0,1]}),
            3: pd.DataFrame({'anchor':['the cat','a dog','the bird','a cat'],
                             'positive':['a cat','the dog','a bird','the cat'],
                             'negative':['a bird','the bird','a dog','the dog']}),
            4: pd.DataFrame({'sentence_1':['the cat','a dog','the bird','a cat'],
                             'sentence_2':['a cat','the bird','a bird','the dog'], 'label':[1,0,1,0]})}
        for tier, folder in [(0,'tang0_style_classifier'),(3,'tang3_triplet_encoder'),(4,'tang4_cross_encoder')]:
            module = importlib.import_module(f'src.models.{folder}.train')
            csv = self.root / f'resume-data-{tier}.csv'
            frames[tier].to_csv(csv, index=False)
            kwargs = dict(train_csv=str(csv), val_csv=str(csv), pretrained_name=str(self.base),
                          max_length=17, batch_size=2, num_epochs=2, seed=71, save_every_steps=1)
            full = self.root / f'resume-full-{tier}'
            module.train(output_dir=str(full), **kwargs)
            expected = torch.load(full / f'tang{tier}_last.pt', weights_only=True)
            for cutoff in (1, 3):
                interrupted = self.root / f'resume-cut-{tier}-{cutoff}'
                original = TrainingRun.after_step
                def cut(run, index, loss):
                    original(run, index, loss)
                    if run.state['global_step'] == cutoff:
                        raise RuntimeError('simulated process kill')
                with patch.object(TrainingRun, 'after_step', cut):
                    with self.assertRaisesRegex(RuntimeError, 'simulated process kill'):
                        module.train(output_dir=str(interrupted), **kwargs)
                checkpoint = torch.load(interrupted / f'tang{tier}_last.pt', weights_only=True)
                self.assertEqual(checkpoint['progress']['next_batch'], 1)
                self.assertEqual(checkpoint['progress']['global_step'], cutoff)
                resumed = self.root / f'resume-result-{tier}-{cutoff}'
                with patch('src.models.runtime.AutoModel.from_pretrained', side_effect=AssertionError('resume must be offline')):
                    module.train(output_dir=str(resumed), resume_from=str(interrupted), **kwargs)
                actual = torch.load(resumed / f'tang{tier}_last.pt', weights_only=True)
                self.assertEqual(actual['progress'], expected['progress'])
                self.assertEqual(actual['history'], expected['history'])
                steps = pd.read_csv(resumed/'steps_history.csv')
                self.assertEqual(steps.global_step.tolist(), [1, 2, 3, 4])
                epochs = pd.read_csv(resumed/'epochs_history.csv')
                self.assertEqual(epochs.epoch.tolist(), [1, 2])
                self.assertTrue(np.isfinite(epochs.val_loss).all())
                self.assertTrue((epochs.val_loss >= 0).all())
                for key, value in expected['model'].items():
                    torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
                for pid, state in expected['optimizer']['state'].items():
                    for key, value in state.items():
                        torch.testing.assert_close(actual['optimizer']['state'][pid][key], value, rtol=0, atol=0)
                for key, value in expected['best_bundle']['state_dict'].items():
                    torch.testing.assert_close(actual['best_bundle']['state_dict'][key], value, rtol=0, atol=0)
                # Không cho resume nhầm batch size hoặc CSV khác.
                with self.assertRaisesRegex(ValueError, 'cấu hình'):
                    module.train(output_dir=str(self.root / f'wrong-config-{tier}-{cutoff}'),
                                 resume_from=str(interrupted), **dict(kwargs, batch_size=1))

    def test_time_budget_saves_before_first_validation_and_resumes(self):
        from src.models.tang4_cross_encoder import train as module
        csv = self.root / 'budget-data.csv'
        pd.DataFrame({'sentence_1':['the cat','the dog'], 'sentence_2':['a cat','a bird'],
                      'label':[1,0]}).to_csv(csv,index=False)
        kwargs = dict(train_csv=str(csv),val_csv=str(csv),pretrained_name=str(self.base),
                      max_length=17,batch_size=1,num_epochs=1)
        output = self.root / 'budget-output'
        result = module.train(output_dir=str(output), max_runtime_minutes=1e-9, **kwargs)
        self.assertIsNone(result)
        saved = torch.load(output/'tang4_last.pt',weights_only=True)
        self.assertEqual(saved['progress']['global_step'],0)
        self.assertFalse((output/'tang4_best.pt').exists())
        module.train(output_dir=str(output),resume_from=str(output), **kwargs)
        self.assertTrue((output/'tang4_best.pt').exists())

    def test_validation_interruption_keeps_completed_batches(self):
        from src.models.tang4_cross_encoder import train as module
        csv = self.root / 'validation-interruption.csv'
        pd.DataFrame({'sentence_1':['the cat','the dog'], 'sentence_2':['a cat','a bird'],
                      'label':[1,0]}).to_csv(csv,index=False)
        kwargs = dict(train_csv=str(csv),val_csv=str(csv),pretrained_name=str(self.base),
                      max_length=17,batch_size=1,num_epochs=1)
        output = self.root / 'validation-interruption'
        with patch.object(module, 'evaluate', side_effect=RuntimeError('validation interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                module.train(output_dir=str(output), **kwargs)
        saved = torch.load(output/'tang4_last.pt',weights_only=True)
        self.assertEqual(saved['progress']['next_batch'],2)
        with patch('torch.optim.AdamW.step', side_effect=AssertionError('already trained batches')):
            module.train(output_dir=str(output), resume_from=str(output), **kwargs)
        self.assertTrue((output/'tang4_best.pt').exists())

    def test_atomic_checkpoint_failed_write_keeps_previous(self):
        from src.models.training_checkpoint import atomic_save
        path = self.root / 'atomic.pt'
        atomic_save({'step': 7}, path)
        def failed_save(payload, stream):
            stream.write(b'partial')
            raise OSError('disk full')
        with patch('torch.save', side_effect=failed_save):
            with self.assertRaises(OSError):atomic_save({'step':8}, path)
        self.assertEqual(torch.load(path, weights_only=True)['step'],7)


if __name__=='__main__':
    unittest.main()
