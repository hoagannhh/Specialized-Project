import unittest
from unittest.mock import Mock

from demo_app.service import _compare_passages


class PairPassageTests(unittest.TestCase):
    def test_partial_match_preserves_offset_and_source_identity(self):
        source = 'Different wording. Original   sentence. Final source.'
        suspect = '🙂 First.  **Reworded   sentence.**\nLast.'
        pipeline = Mock()
        pipeline.closest_source_sentences.return_value = [2, 1, 0]

        def predict(query, candidate):
            hit = (query, candidate) == ('Reworded sentence.', 'Original sentence.')
            return {'probability': 0.9 if hit else 0.1, 'threshold_used': 0.6,
                    'decision': 'suspicious' if hit else 'checked_negative', 'complete': False}

        pipeline.predict.side_effect = predict
        result = _compare_passages(pipeline, suspect, source, lambda _: None)
        self.assertEqual(result['suspicious_sentences'], 1)
        self.assertEqual(result['total_sentences'], 3)
        match = result['results'][1]
        self.assertEqual(match['status'], 'suspicious')
        self.assertEqual(suspect[match['start']:match['end']], 'Reworded   sentence.')
        self.assertEqual(source[match['source_start']:match['source_end']], 'Original   sentence.')
        self.assertEqual(pipeline.predict.call_count, 3)

    def test_repeated_sentence_keeps_each_position_and_reuses_prediction(self):
        pipeline = Mock()
        pipeline.closest_source_sentences.return_value = [0, 0]
        pipeline.predict.return_value = {'probability': 0.1, 'decision': 'checked_negative', 'complete': True}
        result = _compare_passages(pipeline, 'Again. Again.', 'Source.', lambda _: None)
        self.assertEqual([r['start'] for r in result['results']], [0, 7])
        self.assertEqual(result['suspicious_sentences'], 0)
        pipeline.predict.assert_called_once_with('Again.', 'Source.')
