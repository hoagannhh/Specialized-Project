"""Extract body text with stable paragraph offsets; never silently discard bad bytes."""
from pathlib import Path


def iter_docx_paragraphs(parent):
    from docx.document import Document as DocumentType
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.oxml.ns import qn
    container=parent.element.body if isinstance(parent,DocumentType) else parent._tc
    for element in container.iterchildren():
        if element.tag==qn('w:p'):yield Paragraph(element,parent)
        elif element.tag==qn('w:tbl'):
            seen=set()
            for row in Table(element,parent).rows:
                for cell in row.cells:
                    if cell._tc in seen:continue
                    seen.add(cell._tc)
                    yield from iter_docx_paragraphs(cell)


def read_document(file_path):
    path=Path(file_path);suffix=path.suffix.lower();warnings=[];ranges=[]
    if suffix=='.txt':text=path.read_text(encoding='utf-8-sig')
    elif suffix=='.docx':
        import docx
        document=docx.Document(path)
        paragraphs=list(iter_docx_paragraphs(document))
        texts=[p.text for p in paragraphs];text='\n'.join(texts);offset=0
        for item in texts:
            ranges.append({'start':offset,'end':offset+len(item)})
            offset+=len(item)+1
        warnings.append('Phạm vi DOCX: thân bài và bảng; chưa phân tích ảnh, header/footer và footnote.')
    elif suffix=='.pdf':
        import pdfplumber
        pages=[]
        with pdfplumber.open(path) as document:
            for i,page in enumerate(document.pages,1):
                content=page.extract_text() or '';pages.append(content)
                if not content.strip():warnings.append(f'Trang PDF {i} chưa có văn bản đọc được; cần OCR hoặc kiểm tra thủ công.')
        text='\n\n'.join(pages)
    else:raise ValueError('Chỉ hỗ trợ TXT UTF-8, DOCX và PDF có lớp văn bản')
    if not text.strip():raise ValueError('Không có văn bản đọc được. PDF scan cần OCR trước khi kiểm tra.')
    return {'text':text,'file_path':str(path.resolve()),'format':suffix[1:],
            'paragraph_ranges':ranges,'warnings':warnings}


def read_file(file_path):
    return read_document(file_path)['text']
