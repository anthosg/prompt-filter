import base64
import binascii
import re
from dataclasses import dataclass

BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{24,}={0,2})")
HEX_RE = re.compile(r"(?:0x)?[0-9a-fA-F]{32,}")


@dataclass
class EncodingSignals:
    has_base64: bool
    has_hex: bool
    base64_candidates: int
    hex_candidates: int


def _looks_like_decodable_base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
        return True
    except (binascii.Error, ValueError):
        return False


def detect_encoding_signals(text: str) -> EncodingSignals:
    base64_candidates = [m.group(0) for m in BASE64_RE.finditer(text)]
    valid_base64 = [v for v in base64_candidates if _looks_like_decodable_base64(v)]
    hex_candidates = HEX_RE.findall(text)

    return EncodingSignals(
        has_base64=bool(valid_base64),
        has_hex=bool(hex_candidates),
        base64_candidates=len(valid_base64),
        hex_candidates=len(hex_candidates),
    )
