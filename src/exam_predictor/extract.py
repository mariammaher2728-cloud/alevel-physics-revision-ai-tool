"""Step 4 · Read a PDF into question records.

- Question papers are read by layout. The main question number sits in a narrow left column, the labels (a) and (i)
  sit further right, and marks look like [2].
- Mark schemes are read in the PDF's own order, which keeps the answer table rows together.
- Each scored part is matched with its mark-scheme entry.
- Powers keep their meaning (10^5, m s^-1) and subscripts show as c_0. PapaCambridge watermarks and barcodes are removed.
"""
import re

import pymupdf as fitz

from .text_utils import clean_text


JUNK = re.compile(r"©|UCLES|PapaCambridge|Trace ID|Licensed for|Re-uploading|Downloaded from|DO NOT WRITE|"
                  r"\[Turn over|BLANK PAGE|Section [AB]|^\*\s*\d{6,}|\bDFD\b|9702/\d\d/", re.I)


def line_text(line):
    """Join the pieces (spans) of one PDF line, marking powers with ^ and subscripts with _ (10^5, c_0)."""
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return ""
    main = max(spans, key=lambda s: s["size"] * len(s["text"]))
    out = ""
    for s in line["spans"]:
        t, small = s["text"], s["size"] < 0.85 * main["size"]
        if t.strip() and (s["flags"] & 1 or (small and s["origin"][1] < main["origin"][1] - 1)):
            t = "^" + t.lstrip()
        elif t.strip() and small and s["origin"][1] > main["origin"][1] + 1:
            t = "_" + t.lstrip()
        out += t
    return out


def page_lines(page, pno):
    """Question-paper lines in reading order, without headers, footers, margins and watermarks."""
    W, H = page.rect.width, page.rect.height
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            x, y = line["bbox"][:2]
            text = clean_text(line_text(line))
            if (not text or not 0.05 * H < y < 0.94 * H or x < 0.04 * W or abs(line["dir"][1]) > 0.1
                    or text in {"For", "Examiner’s", "Examiner's", "Use"} or JUNK.search(text)
                    or sum(ord(c) < 32 for c in text) > 3
                    or sum("Ā" <= c <= "ſ" for c in text) > 2):    # barcode gibberish
                continue
            lines.append({"text": text, "x": x / W, "y": y, "page": pno})
    return sorted(lines, key=lambda l: (round(l["y"] / 3), l["x"]))


def extract_questions(path):
    """One record per scored part, keeping the parent question's shared context."""
    groups, current = {}, None
    with fitz.open(path) as doc:
        for pno in range(2, len(doc) + 1):                       # page 1 is the cover
            for line in page_lines(doc[pno - 1], pno):
                m = re.fullmatch(r"(\d{1,2})\s*(.*)", line["text"])
                if m and 0.075 <= line["x"] <= 0.095 and 1 <= int(m[1]) <= 25:   # main question number
                    number = int(m[1])
                    if current is None and number != 1:
                        continue
                    if current is None or number == current + 1:
                        current = number
                        groups[current] = [{**line, "text": m[2], "x": 0.12}] if m[2] else []
                        continue
                if current is not None:
                    groups[current].append(line)
    records = []
    for number, lines in groups.items():
        nodes, path_ = {(): []}, ()                              # () holds the parent question's context
        for line in lines:
            label = re.match(r"^\(([a-z]+)\)\s*(.*)", line["text"])
            if label and line["x"] < 0.21:
                token, rest = label.groups()
                if len(token) == 1 and line["x"] < 0.145:
                    path_ = (token,)
                    nested = re.match(r"^\(([ivx]+)\)\s*(.*)", rest)
                    if nested:
                        path_ += (nested[1],)
                elif re.fullmatch(r"[ivx]+", token) and path_:
                    path_ = (path_[0], token)
            nodes.setdefault(path_, []).append(line)
        full_text = "\n".join(l["text"] for l in lines)
        for path_, body in nodes.items():
            text = "\n".join(l["text"] for l in body)
            marks = re.findall(r"\[\s*(\d{1,2})\s*\]", text)
            if not marks:
                continue
            context = [l for depth in range(len(path_)) for l in nodes.get(path_[:depth], [])]
            records.append({"part": str(number) + "".join(f"({p})" for p in path_), "question_number": number,
                            "text": text, "marks": sum(map(int, marks)),
                            "context": "\n".join(l["text"] for l in context), "full_question": full_text,
                            "context_pages": sorted({l["page"] for l in context + body}),
                            "damaged": bool(re.search(r"[\x00-\x08\x0b-\x1f-]", full_text))})
    return records, sorted(groups)


MS_JUNK = re.compile(r"©|UCLES|Cambridge|PUBLISHED|Syllabus|Teachers|Page \d|9702|May/June|March 20|October/November|"
                     r"PapaCambridge|Trace ID|Licensed for|Re-uploading|Downloaded from|Section [AB]|\bDFD\b|^\*\s*\d{6,}", re.I)


def extract_mark_scheme(path):
    """Mark-scheme text grouped by part label, e.g. {'1(b)(i)': {...}}."""
    entries, current, question, letter = {}, None, None, None
    with fitz.open(path) as doc:
        for pno in range(2, len(doc) + 1):
            page = doc[pno - 1]
            blocks = page.get_text("dict")["blocks"]
            main_labels = {}                                     # main question numbers in the left column
            for block in blocks:
                for line in block.get("lines", []):
                    token = clean_text(line_text(line))
                    box = fitz.Rect(line["bbox"]) * page.rotation_matrix
                    start = re.match(r"^(\d{1,2})(?:\s+(.*)|$)", token)
                    if start and box.x0 < 0.12 * page.rect.width and 1 <= int(start[1]) <= 25:
                        main_labels[token] = (start[1], start[2] or "")
            for block in blocks:                                 # the PDF's own order keeps table rows together
                for line in block.get("lines", []):
                    text = re.sub(r"^(\d{1,2})([a-z])(\([ivx]+\))(?=\s|$)", r"\1(\2)\3", clean_text(line_text(line)))
                    if not text or MS_JUNK.search(text) or text in {"Question", "Answer", "Marks", "Paper", "Mark Scheme"}:
                        continue
                    full = re.match(r"^(\d{1,2})\s*(\([a-z]\)(?:\s*\([ivx]+\))?)\s*(.*)", text)
                    if full:
                        question, letter = full[1], re.findall(r"\(([a-z])\)", full[2])[0]
                        current, text = question + re.sub(r"\s+", "", full[2]), full[3]
                    elif text in main_labels and int(main_labels[text][0]) > int(question or 0):
                        question, text = main_labels[text]
                        letter, current = None, question
                    elif question and re.match(r"^\([a-z]+\)", text):
                        token, text = re.match(r"^\(([a-z]+)\)\s*(.*)", text).groups()
                        if len(token) == 1 and (token not in {"i", "v", "x"} or letter is None):
                            letter, current = token, f"{question}({token})"
                            nested = re.match(r"^\(([ivx]+)\)\s*(.*)", text)
                            if nested:
                                current, text = current + f"({nested[1]})", nested[2]
                        elif letter and re.fullmatch(r"[ivx]+", token):
                            current = f"{question}({letter})({token})"
                    if current:
                        entry = entries.setdefault(current, {"lines": [], "pages": set()})
                        if text:
                            entry["lines"].append(text)
                        entry["pages"].add(pno)
    return {k: {"text": "\n".join(v["lines"]), "pages": sorted(v["pages"])} for k, v in entries.items()}


def extract_pair(qp, ms):
    """Read one paper, attach mark-scheme answers, and run simple structure checks."""
    records, numbers = extract_questions(qp)
    answers = extract_mark_scheme(ms)
    for r in records:
        part = r["part"]
        answer, match = answers.get(part), "exact"
        if not answer:                                           # e.g. the scheme only has 2(b)(i) and 2(b)(ii)
            kids = [v for k, v in answers.items() if k.startswith(part + "(")]
            answer = {"text": "\n".join(v["text"] for v in kids), "pages": sorted({p for v in kids for p in v["pages"]})} if kids else None
            match = "children" if kids else "missing"
        while answer is None and "(" in part:                    # fall back to the parent entry
            part = part[:part.rfind("(")]
            if part in answers:
                answer, match = answers[part], "parent"
        r.update(answer=answer["text"] if answer else "", answer_pages=answer["pages"] if answer else [], answer_match=match)
    total = sum(r["marks"] for r in records)
    ok = bool(numbers) and numbers == list(range(1, max(numbers) + 1)) and total == 100 \
         and len({r["part"] for r in records}) == len(records)
    return records, {"extraction_ok": ok, "marks": total, "questions": len(numbers), "parts": len(records),
                     "missing_answers": sum(r["answer_match"] == "missing" for r in records)}
