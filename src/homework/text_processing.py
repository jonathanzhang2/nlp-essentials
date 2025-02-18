import re
from typing import Optional


def chronicles_of_narnia(path: str) -> dict:
    text = ''.join(open(file=path, mode='r').readlines())
    out: dict[str, dict[str, str | int | list[dict[str, int | str]]]] = dict()
    
    book = re.compile(pattern=r'\b(.+)(?:\s\(\s)(\d{4})(?:\s\)\n)')
    chapter = re.compile(pattern=r'(?:\b[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s[IVXLCDM]+\n)')
    title = re.compile(pattern=r'(.+?)(?:\n)(.+)', flags=re.DOTALL)
    
    for match in book.finditer(string=text):
        out[match.group(1)] = {
            'title': match.group(1),
            'year': int(match.group(2)),
            'chapters': list()
        }
        current_book = (
            text[match.span()[1]:match.span()[1]+next_book.span()[0] 
                if (next_book := book.search(string=text[match.span()[1]:])) 
            else len(text)].rstrip('\n')
        )
        for i, match_ch in enumerate(chapter.finditer(string=current_book)):
            current_chapter = (
                current_book[match_ch.span()[1]:match_ch.span()[1]+next_chapter.span()[0] 
                    if (next_chapter := chapter.search(string=current_book[match_ch.span()[1]:])) 
                else len(current_book)].rstrip('\n')
            )
            out[match.group(1)]['chapters'].append({
                'number': i + 1,
                'title': (match_text := title.search(string=current_chapter)).group(1),
                'token_count': len(match_text.group(2).split())
            })
    return out
    

def regular_expressions(string: str) -> Optional[str]:
    matches: dict[str, re.Pattern] = {
        'email': re.compile(pattern=r'(?:(?:[a-zA-Z0-9]+[\w.-]+[a-zA-Z0-9]+)|(?:[a-zA-Z0-9]?))@(?:(?:[a-zA-Z0-9]+[\w.-]+[a-zA-Z0-9]+)|(?:[a-zA-Z0-9]?))\.(?:com|org|edu|gov)'),
        'date': re.compile(pattern=r'(?:(?:(?:(?:19)?(?:5[1345789]|[79][01345789]|[68][1235679]))|(?:(?:20)?(?:[024][1235679]|[13][01345789])))([/-]{1})(?:(?:(?:(?:1[02])|(?:0?[13578]))\1(?:(?:3[01])|(?:2[0-9])|(?:1[0-9])|(?:0?[1-9])))|(?:(?:(?:11)|(?:0?[469]))\1(?:(?:30)|(?:2[0-9])|(?:1[0-9])|(?:0?[1-9])))|(?:0?2\1(?:(?:2[0-8])|(?:1[0-9])|(?:0?[1-9])))))|(?:(?:(?:(?:19)?(?:[579][26]|[68][048]))|(?:(?:20)?(?:50|[024][048]|[13][26])))([/-]{1})(?:(?:(?:(?:1[02])|(?:0?[13578]))\2(?:(?:3[01])|(?:2[0-9])|(?:1[0-9])|(?:0?[1-9])))|(?:(?:(?:11)|(?:0?[469]))\2(?:(?:30)|(?:2[0-9])|(?:1[0-9])|(?:0?[1-9])))|(?:0?2\2(?:(?:2[0-9])|(?:1[0-9])|(?:0?[1-9])))))'),
        'url': re.compile(pattern=r'https?://[a-zA-Z0-9](?:[a-zA-Z-.])*?.+(?:[a-zA-Z-.])*'),
        'cite': re.compile(pattern=r'[A-Z][a-zA-Z]*(?:[\'-][A-Z][a-zA-Z]*)?(?:\set\sal\.|\sand\s[A-Z][a-zA-Z]*(?:[\'-][A-Z][a-zA-Z]*)?)?,\s(?:(?:19[0-9]{2})|(?:20(?:[01][0-9]|2[0-4])))')
    }
    return types[0] if any(types := [k for k, v in matches.items() if v.fullmatch(string=string)]) else None
