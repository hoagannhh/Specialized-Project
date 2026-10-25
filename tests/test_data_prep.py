import json
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path
import pandas as pd
from src.data_prep.prepare_datasets import build, prepare_sources, Components, read_padben
from src.data_prep.build_triplet import sample_negatives, make_triplets
from src.data_prep.audit_datasets import audit


@contextmanager
def test_directory():
    # Use workspace fixtures; retained for debugging failed tests.
    path = Path(__file__).resolve().parents[1] / 'tmp' / ('data-prep-test-' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield str(path)


def fixture(path):
    originals = [{'text': f'Original topic number {i}',
                  'paraphrases': repr([f'Reworded topic number {i}', f'Alternate topic number {i}'])}
                 for i in range(100)]
    # Transitive overlap, including changed whitespace and human/AI label conflict.
    originals += [{'text': 'Another origin', 'paraphrases': repr(['Reworded  topic number 0'])},
                  {'text': 'Reworded topic number 1', 'paraphrases': repr(['Yet another variant'])}]
    pd.DataFrame(originals).to_csv(path / 'humarin_chatgpt_paraphrases.csv', index=False)
    data = {t: [] for t in range(1,6)}
    for i in range(30):
        h, hp, ai, ap, first, deep = [f'{kind} example {i}' for kind in ['human','humanpara','generated','aipara','first','deep']]
        if i == 1:
            deep = h  # conflicting label only visible in held-out task 4/5
        if i == 0:
            h = 'Original topic number 0'  # overlap across datasets
        pairs = {1: [hp, ap], 2: [h, ai], 3: [ap, first], 4: [first, deep], 5: [h, deep]}
        for t in data:
            # Vary order and IDs: grouping must use content, not matching idx.
            texts, labels = (pairs[t][::-1], [1,0]) if i % 2 else (pairs[t], [0,1])
            data[t].append({'idx': i + t * 100, 'sentence_pair': texts, 'label_pair': labels})
    for t, rows in data.items():
        (path / f'padben_sentence_pair_task{t}.json').write_text(json.dumps(rows), encoding='utf-8')


class DataPrepTests(unittest.TestCase):
    def test_negatives_all_strategies_small_and_last_singleton(self):
        df = pd.DataFrame({'group_id': [0,0,0,1,2],
                           'original_text': ['common apple']*3 + ['common pear','common banana'],
                           'paraphrase_text': ['common red','common green','common fruit','common yellow','common ripe']})
        for strategy in ['random','hard','mixed']:
            idx, kinds = sample_negatives(df, strategy, chunk_size=2, top_k_pool=50)
            result = make_triplets(df, idx, kinds)
            self.assertTrue((result.group_id != result.negative_group_id).all())
        with self.assertRaises(ValueError):
            sample_negatives(df.iloc[:3])

    def test_padben_mapping_and_reversed_positions(self):
        with test_directory() as temp:
            path = Path(temp); fixture(path)
            pad = read_padben(path, Components())
            self.assertEqual(set(pad[pad.task.isin([3,4])].label), {1})
            for task in [1,2,5]:
                subset = pad[pad.task == task]
                self.assertTrue((subset.label == subset.original_task_label).all())

    def test_end_to_end_shared_split_and_conflicts(self):
        with test_directory() as temp:
            path = Path(temp); fixture(path)
            pairs, singles, pad, manifest, conflicts, stats = prepare_sources(path)
            self.assertIn('Reworded topic number 1', set(conflicts.sentence))
            self.assertIn('human example 1', set(conflicts.sentence))
            self.assertNotIn('Reworded topic number 1', set(singles.sentence))
            related = pairs[pairs.original_text.isin(['Original topic number 0', 'Another origin'])]
            self.assertEqual(related.group_id.nunique(), 1)
            out = path / 'out'
            build(path, out, 'mixed', chunk_size=7)
            self.assertEqual(audit(out)['cross_split_text_overlap'], 0)
            # Repeated runs produce the same split manifest.
            _, _, _, again, _, _ = prepare_sources(path)
            pd.testing.assert_frame_equal(manifest, again)
            # The independent audit must reject an exported cross-split text.
            train = pd.read_csv(out / 'tang0_train.csv')
            test = pd.read_csv(out / 'tang0_test.csv')
            test.loc[0, 'sentence'] = train.iloc[0].sentence
            test.to_csv(out / 'tang0_test.csv', index=False)
            with self.assertRaises(ValueError):
                audit(out)


if __name__ == '__main__':
    unittest.main()
