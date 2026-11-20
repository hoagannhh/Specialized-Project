import os
import unittest
from unittest.mock import Mock, patch

from google import genai
from google.genai import errors
from src.models.tang5_llm_judge import call_api as judge


class GeminiQuotaTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'GEMINI_API_KEYS': 'test-a,test-b,test-a'}))
        self.enterContext(patch.object(judge, '_gemini_active_key', {}))
        self.enterContext(patch.object(judge, '_gemini_retry_at', {}))
        self.clock = self.enterContext(patch.object(judge, 'monotonic', return_value=100))
        self.client = self.enterContext(patch.object(genai, 'Client'))
        self.generate = self.client.return_value.models.generate_content
        self.ok = Mock(text='ok')
        self.quota = errors.ClientError(429, {'error': {
            'status': 'RESOURCE_EXHAUSTED',
            'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '15s'}],
        }})

    def used_keys(self):
        return [call.kwargs['api_key'] for call in self.client.call_args_list]

    def test_retry_same_prompt_then_keep_working_key(self):
        self.generate.side_effect = [self.ok, self.ok, self.quota, self.ok, self.ok]
        for prompt in ['first', 'second', 'third', 'fourth']:
            self.assertEqual(judge.call_gemini(prompt), 'ok')
        self.assertEqual(self.used_keys(), ['test-a', 'test-a', 'test-a', 'test-b', 'test-b'])
        self.assertEqual([c.kwargs['contents'] for c in self.generate.call_args_list],
                         ['first', 'second', 'third', 'third', 'fourth'])

    def test_all_keys_limited_then_recover(self):
        self.generate.side_effect = [self.quota, self.quota, self.ok]
        with self.assertRaises(judge.GeminiQuotaExhausted):
            judge.call_gemini('first')
        self.clock.return_value = 114
        with self.assertRaises(judge.GeminiQuotaExhausted):
            judge.call_gemini('second')
        self.assertEqual(self.used_keys(), ['test-a', 'test-b'])
        self.clock.return_value = 115
        self.assertEqual(judge.call_gemini('third'), 'ok')

    def test_other_errors_and_explicit_key(self):
        for code in [400, 403, 503]:
            with self.subTest(code=code):
                self.client.reset_mock()
                self.generate.side_effect = errors.APIError(code, {})
                with self.assertRaises(errors.APIError):
                    judge.call_gemini('check')
                self.assertEqual(self.used_keys(), ['test-a'])
        self.client.reset_mock()
        self.generate.side_effect = self.quota
        with self.assertRaises(errors.ClientError):
            judge.call_gemini('check', api_key='explicit-test-key')
        self.assertEqual(self.used_keys(), ['explicit-test-key'])
