"""Kiểm tra đầu vào, báo cáo và chặn pipeline chưa đủ model; không gọi API."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from demo_app import service


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=service.ROOT/'tmp')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_uploaded_name_cannot_choose_output_path(self):
        doc=service.document_input({'file':{'name':'../../outside.txt','data':base64.b64encode(b'Hello world.').decode()}},self.root,'input')
        self.assertEqual(Path(doc['file_path']),self.root/'input.txt')
        self.assertEqual(doc['text'],'Hello world.')

    def test_invalid_inputs_are_rejected(self):
        for value in [{}, {'file':{'name':'run.exe','data':'AA=='}}, {'file':{'name':'a.txt','data':'!bad!'}}, {'text':'x'*40001}]:
            with self.assertRaises(ValueError):service.document_input(value,self.root,'input')

    def test_full_mode_does_not_fallback_to_tier3(self):
        with patch.object(service,'OUTPUT',self.root),patch.object(service,'capabilities',return_value={'full_pair':False}):
            with self.assertRaisesRegex(ValueError,'Chưa đủ'):
                service.analyze({'mode':'full','kind':'pair'},'job',lambda _:None)

    def test_empty_placeholder_is_not_a_model(self):
        folder=self.root/'tang3';folder.mkdir()
        for name in ['tang3_best.pt','model_config.json','tokenizer_config.json','tokenizer.json']:(folder/name).touch()
        with patch.object(service.config,'CHECKPOINTS_DIR',self.root):
            self.assertFalse(service.bundle_status(3)['ready'])

    def test_demo_report_escapes_text_and_has_word_highlight(self):
        text='<script>alert(1)</script>.'
        result={'original_text':text,'threshold':.75,'warnings':[service.NOTICE],
                'results':[{'start':0,'end':len(text),'sentence':text,'sentence_id':0,
                            'highlight':True,'similarity':.9,'matches':[{'source_text':'Original text.'}]}]}
        service.export_demo(result,self.root)
        html=(self.root/'report.html').read_text(encoding='utf8')
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        from docx import Document
        word=Document(self.root/'report.docx')
        self.assertTrue(any(r.font.highlight_color is not None for p in word.paragraphs for r in p.runs))


if __name__=='__main__':unittest.main()
