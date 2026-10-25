"""Independent streaming checks on exported model inputs (all text roles)."""
import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path


def audit(directory):
    directory = Path(directory)
    with (directory / 'source_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        groups = {}
        for row in csv.DictReader(f):
            old = groups.setdefault(row['group_id'], row['split'])
            if old != row['split']:
                raise ValueError('A source component spans splits')
    owners = {}
    sentence_labels = {}
    counts = {}
    for split in ['train', 'val', 'test']:
        specs = [('tang0', ['sentence'], ['group_id']),
                 ('triplet_data', ['anchor', 'positive', 'negative'], ['group_id', 'negative_group_id']),
                 ('pairwise_labeled', ['sentence_1', 'sentence_2'], ['group_id', 'candidate_group_id'])]
        if split == 'test':
            specs += [('padben_task4', ['sentence'], ['group_id']),
                      ('padben_task5', ['sentence'], ['group_id'])]
        for prefix, columns, group_cols in specs:
            path = directory / f'{prefix}_{split}.csv'
            count = 0
            with path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    count += 1
                    for col in group_cols:
                        if groups.get(row[col]) != split:
                            raise ValueError(f'Wrong group ownership: {path.name}:{count}')
                    if prefix == 'triplet_data':
                        if row['group_id'] == row['negative_group_id'] or row['negative'] in (row['anchor'], row['positive']):
                            raise ValueError(f'Invalid triplet: {path.name}:{count}')
                    elif prefix == 'pairwise_labeled':
                        same = row['group_id'] == row['candidate_group_id']
                        if (row['label'] == '1') != same or row['label'] not in ('0','1'):
                            raise ValueError(f'Invalid pair provenance: {path.name}:{count}')
                    elif row['label'] not in ('0', '1'):
                        raise ValueError(f'Invalid sentence label: {path.name}:{count}')
                    for col in columns:
                        text = ' '.join(unicodedata.normalize('NFC', row[col]).split())
                        if not text:
                            raise ValueError(f'Empty text: {path.name}:{count}')
                        digest = hashlib.sha256(text.encode()).digest()
                        if columns == ['sentence']:
                            previous_label = sentence_labels.setdefault(digest, row['label'])
                            if previous_label != row['label']:
                                raise ValueError(f'Contradictory sentence labels: {path.name}:{count}')
                        old = owners.setdefault(digest, split)
                        if old != split:
                            raise ValueError(f'Text crosses {old}/{split}: {path.name}:{count}')
            counts[path.name] = count
            print(f'Audit OK: {path.name} ({count:,} rows)', flush=True)
    report = {'status': 'passed', 'cross_split_text_overlap': 0,
              'cross_split_group_overlap': 0, 'unique_texts': len(owners), 'rows': counts}
    (directory / 'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
