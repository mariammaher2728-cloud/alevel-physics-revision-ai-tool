"""Small text helpers used by several steps."""
import re
import unicodedata


def sitting_label(idx):
    """Turn a sitting number (year*3 + session) back into a label such as 'MJ 2024'."""
    return f"{['FM', 'MJ', 'ON'][idx % 3]} {idx // 3}"


def clean_text(text):
    """One standard form for each character, no dotted answer lines, single spaces."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\.{4,}|…{2,}", " ", text)).strip()


def squash(text, n=None):
    """Put the text on one line (and cut it to n characters)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:n] if n else text


def part_text(context, text, answer):
    """The text that represents one part: end of its context + the question + start of its mark scheme."""
    return f"Context: {squash(context)[-400:]}\nQuestion: {squash(text)}\nMark scheme: {squash(answer, 400)}"


COMMANDS = {"show that": "Show that", "state what is meant by": "Define", "define": "Define", "state": "State",
            "explain": "Explain", "describe": "Describe", "suggest": "Suggest", "calculate": "Calculate",
            "determine": "Determine", "derive": "Derive", "deduce": "Deduce", "estimate": "Estimate",
            "sketch": "Sketch", "draw": "Draw", "complete": "Complete", "compare": "Compare", "outline": "Outline",
            "identify": "Identify", "predict": "Predict", "discuss": "Discuss", "plot": "Plot"}
COMMAND_RE = re.compile(r"\b(" + "|".join(k.replace(" ", r"\s+") for k in sorted(COMMANDS, key=len, reverse=True)) + r")\b",
                        re.I)


def command_word(text, context=""):
    """The Cambridge command word of a part. For list parts such as '(i) the radius', it is taken from the stem ('Calculate:')."""
    body = re.sub(r"^\s*(\(\w+\)\s*)+", "", str(text))[:200]
    m = COMMAND_RE.search(body) or COMMAND_RE.search(str(context)[-150:])
    return COMMANDS[" ".join(m[1].lower().split())] if m else "Other"
