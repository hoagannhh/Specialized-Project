"""Notebook structure and shared IO contracts; no downloads or full training."""
import json
from pathlib import Path
import unittest
from uuid import uuid4
import nbformat
import pandas as pd
from notebooks.workflow import inspect_csv,run_script,require_checkpoint

ROOT=Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_all_six_notebooks_valid_clean_and_compilable(self):
        paths=list((ROOT/'notebooks').glob('*.ipynb'))
        self.assertEqual(len(paths),6)
        for path in paths:
            notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
            self.assertTrue(any(c.cell_type=='markdown' for c in notebook.cells))
            for index,cell in enumerate(notebook.cells):
                if cell.cell_type=='code':
                    compile(cell.source,f'{path.name}:{index}','exec')
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs,[])

    def test_stream_summary_and_sampling_are_repeatable(self):
        root=ROOT/'tmp'/('notebook-test-'+uuid4().hex);root.mkdir(parents=True)
        path=root/'data.csv'
        pd.DataFrame({'sentence':[f'text {i}' for i in range(100)],'label':[0,1]*50,'source':['example']*100}).to_csv(path,index=False)
        summary,sample=inspect_csv(path,['sentence'],7)
        self.assertEqual(summary['rows'],100);self.assertEqual(summary['labels'],{0:50,1:50})
        self.assertEqual(len(sample),7)
        pd.testing.assert_frame_equal(sample,inspect_csv(path,['sentence'],7)[1])
        self.assertNotEqual(sample.sentence.tolist(),[f'text {i}' for i in range(7)])

    def test_disabled_command_does_not_run_and_failures_propagate(self):
        root=ROOT/'tmp'/('notebook-test-'+uuid4().hex);root.mkdir(parents=True)
        script=root/'fail.py';script.write_text('raise SystemExit(3)',encoding='utf-8')
        run_script(root,'fail.py',[],enabled=False)
        with self.assertRaises(RuntimeError):run_script(root,'fail.py',[],enabled=True,log_path=root/'log.txt')
        with self.assertRaises(FileNotFoundError):require_checkpoint(root,0)

    def test_detailed_profile_counts_full_data_not_sample(self):
        from notebooks.eda_tools import profile_csv
        root=ROOT/'tmp'/('eda-test-'+uuid4().hex);root.mkdir(parents=True)
        path=root/'input.csv'
        pd.DataFrame({'sentence':['hello world','   ',None,'one two three'],
                      'label':[0,1,1,9],'source':['s']*4}).to_csv(path,index=False)
        report,sample=profile_csv(path,['sentence'],sample_size=1)
        self.assertEqual(report['rows'],4);self.assertEqual(len(sample),1)
        self.assertEqual(report['blank_including_missing']['sentence'],2)
        self.assertEqual(report['missing']['sentence'],1)
        self.assertEqual(report['invalid_labels'],1)
        self.assertEqual(report['word_lengths'][0]['p95_words'],3)

    def test_disk_audit_detects_leakage_and_conflicting_labels(self):
        from notebooks.eda_tools import disk_overlap_audit
        root=ROOT/'tmp'/('eda-test-'+uuid4().hex);root.mkdir(parents=True)
        specs=[]
        for split,label in [('train',0),('val',1)]:
            path=root/f'{split}.csv'
            pd.DataFrame({'sentence':['same  text','same text'],'label':[label,label],'group_id':['shared','shared']}).to_csv(path,index=False)
            specs.append((path,split,['sentence'],['group_id']))
        result=disk_overlap_audit(specs,root/'audit.sqlite')
        self.assertEqual(result,{'cross_split_texts':1,'cross_split_groups':1,'conflicting_single_labels':1,'status':'failed'})
        self.assertEqual(disk_overlap_audit(specs[:1],root/'clean.sqlite')['status'],'passed')


if __name__=='__main__':unittest.main()
