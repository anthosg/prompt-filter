ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space
}


def remove_zero_width(text: str) -> str:
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")
    return text


def count_zero_width(text: str) -> int:
    return sum(text.count(char) for char in ZERO_WIDTH_CHARS)
