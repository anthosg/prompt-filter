import re
from typing import List

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)
