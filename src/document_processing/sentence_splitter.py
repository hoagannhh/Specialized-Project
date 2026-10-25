"""Heuristic sentence boundaries with original character offsets and no short-text loss."""
import re
DEFAULT_MIN_LENGTH=1
_ABBREVIATIONS={'mr.','mrs.','ms.','dr.','prof.','ts.','ths.','pgs.','tp.','e.g.','i.e.','v.v.'}
_BOUNDARY=re.compile(r'(?<=[.!?…])\s+|(?<=[”"\x27])\s+|\n[ \t]*\n+')


def split_sentences_with_position(text,min_length=DEFAULT_MIN_LENGTH):
    if min_length<1:raise ValueError('min_length must be positive')
    cuts=[]
    for match in _BOUNDARY.finditer(text):
        prefix=text[max(0,match.start()-100):match.start()]
        last=prefix.rsplit(None,1)[-1] if prefix.strip() else ''
        blank_line='\n' in match.group() and match.group().count('\n')>=2
        if not blank_line:
            if last.lower() in _ABBREVIATIONS or re.fullmatch(r'[A-Z]\.',last):continue
            if prefix.endswith(('"',"'",'”')) and not re.search(r'[.!?…][”"\x27]$',prefix):continue
        cuts.append((match.start(),match.end()))
    results=[];start=0
    for end,next_start in cuts+[(len(text),len(text))]:
        raw=text[start:end]
        left=len(raw)-len(raw.lstrip());right=len(raw.rstrip())
        sentence=' '.join(raw.split())
        if len(sentence)>=min_length:
            results.append({'sentence':sentence,'start':start+left,'end':start+right})
        start=next_start
    return results


def split_sentences(text,min_length=DEFAULT_MIN_LENGTH):
    return [item['sentence'] for item in split_sentences_with_position(text,min_length)]
