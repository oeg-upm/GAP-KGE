"""Table context + entity extraction (LightOnOCR + GLiNER2).

Single PDF:
    python extract_table_context_lightonocr_gliner.py -i path/to/paper.pdf -o path/to/out -v

Multiple PDFs (folder):
    python extract_table_context_lightonocr_gliner.py -i path/to/pdfs -o path/to/out -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pypdfium2 as pdfium
import torch
from bs4 import BeautifulSoup
from PIL import Image

OCR_MODEL_ID = "lightonai/LightOnOCR-2-1B"
OCR_MAX_NEW_TOKENS = 8192
OCR_TARGET_LONGEST = 1540

GLINER2_MODEL_ID = "fastino/gliner2-base-v1"
GLINER2_MIN_SCORE = 0.65
GLINER2_MAX_CHARS = 3000

ADJACENT_PARA_WINDOW = 2
MIN_MENTION_CHARS = 40

_HEADING_ONLY_RE = re.compile(r"^#{1,6}\s+\S")

GLINER2_LABEL_DESCRIPTIONS = {
    "model": (
        "Name of a model, method, or algorithm "
        "(e.g. TransE, ComplEx, RotatE)."
    ),
    "dataset": (
        "Name of a benchmark dataset or knowledge-graph corpus "
        "(e.g. WN18, FB15k, YAGO)."
    ),
    "metric": (
        "Name of an evaluation metric used for reporting performance "
        "(e.g. MRR, Hits@10, F1)."
    ),
}
GLINER2_LABELS = list(GLINER2_LABEL_DESCRIPTIONS.keys())

log = logging.getLogger(__name__)

# Runtime model handles (set by load_models).
ocr_processor = None
ocr_model = None
ocr_device: str = "cpu"
ocr_dtype = torch.float32
gliner2_model = None


# ---------------------------------------------------------------------------
# Torch / GLiNER environment (DeBERTa JIT workaround)
# ---------------------------------------------------------------------------


def _configure_torch_for_gliner() -> None:
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
    for name, args in [
        ("_jit_set_profiling_executor", (False,)),
        ("_jit_set_profiling_mode", (False,)),
        ("_jit_override_can_fuse_on_gpu", (False,)),
        ("_jit_override_can_fuse_on_cpu", (False,)),
        ("_jit_set_texpr_fuser_enabled", (False,)),
        ("_jit_set_nvfuser_enabled", (False,)),
    ]:
        fn = getattr(torch._C, name, None)
        if fn is not None:
            try:
                fn(*args)
            except Exception:
                pass


def load_models() -> None:
    """Load LightOnOCR and GLiNER2 once."""
    global ocr_processor, ocr_model, ocr_device, ocr_dtype, gliner2_model

    if ocr_model is not None and gliner2_model is not None:
        return

    from gliner2 import GLiNER2
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    if torch.cuda.is_available():
        ocr_device = "cuda"
        ocr_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        ocr_device = "mps"
        ocr_dtype = torch.float16
    else:
        ocr_device = "cpu"
        ocr_dtype = torch.float32

    log.info("OCR device: %s | dtype: %s", ocr_device, ocr_dtype)
    ocr_processor = LightOnOcrProcessor.from_pretrained(OCR_MODEL_ID)
    ocr_model = LightOnOcrForConditionalGeneration.from_pretrained(
        OCR_MODEL_ID,
        torch_dtype=ocr_dtype,
        attn_implementation="eager",
    ).to(ocr_device)

    _configure_torch_for_gliner()
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    gliner2_model = GLiNER2.from_pretrained(GLINER2_MODEL_ID, map_location=map_location)
    log.info("GLiNER2 loaded: %s on %s", GLINER2_MODEL_ID, map_location)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def render_pdf_page(pdf_doc, page_idx: int, target_longest: int = OCR_TARGET_LONGEST) -> Image.Image:
    page = pdf_doc[page_idx]
    img = page.render(scale=200 / 72).to_pil()
    w, h = img.size
    longest = max(w, h)
    if longest > target_longest:
        ratio = target_longest / longest
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img.convert("RGB") if img.mode != "RGB" else img


def ocr_page(img: Image.Image, max_new_tokens: int = OCR_MAX_NEW_TOKENS) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    try:
        conv = [{"role": "user", "content": [{"type": "image", "url": tmp.name}]}]
        inputs = ocr_processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=ocr_device, dtype=ocr_dtype) if v.is_floating_point() else v.to(ocr_device)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1] :]
        return ocr_processor.decode(gen, skip_special_tokens=True)
    finally:
        os.unlink(tmp.name)


def ocr_pdf_pages(
    pdf_path: Path,
    cache_dir: Path,
    *,
    force_ocr: bool = False,
) -> List[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{pdf_path.stem}_pages.json"
    if cache_file.exists() and not force_ocr:
        log.debug("OCR cache hit: %s", cache_file)
        with cache_file.open(encoding="utf-8") as f:
            return json.load(f)["pages"]

    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    pages: List[str] = []
    try:
        n_pages = len(pdf_doc)
        for page_idx in range(n_pages):
            log.info("  OCR page %d/%d", page_idx + 1, n_pages)
            pages.append(ocr_page(render_pdf_page(pdf_doc, page_idx)))
    finally:
        pdf_doc.close()

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump({"file_name": str(pdf_path), "pages": pages}, f, ensure_ascii=False)
    return pages


# ---------------------------------------------------------------------------
# Context patterns
# ---------------------------------------------------------------------------

_NUM = r"[A-Za-z]?\d+(?:\.\d+)?|[IVXLC]+"

TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
CAPTION_RE = re.compile(
    rf"(?:\*\*)?(?:Table|Tab\.?|TABLE)\s+({_NUM})(?:\*\*)?\s*[:.\u2014-]\s+",
    re.IGNORECASE,
)
TABLE_REF_RE = re.compile(
    rf"\b(?:Table|Tab\.?|TABLE|Tables|TABLES)\s+({_NUM})(?:\s*(?:,|and|&)\s*({_NUM}))*",
    re.IGNORECASE,
)
_REF_KEYWORD_RE = re.compile(r"^(?:Tables?|Tab\.?|TABLES?)\s*", re.IGNORECASE)
_REF_NUM_RE = re.compile(r"[A-Za-z]?\d+(?:\.\d+)?|[IVXLC]+")


def normalize_table_number(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def _collapse_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _repair_hyphen_breaks(text: str) -> str:
    """Rejoin hyphenated words split across line breaks (lowercase continuation only)."""
    return re.sub(r"(\w)-\s*\n\s*([a-z][\w'-]*)", r"\1\2", text)


def _split_paragraphs(text: str) -> List[str]:
    """Split page text into paragraphs; fall back to line/sentence chunks when OCR has no blank lines."""
    text = _repair_hyphen_breaks(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) > 1:
        return [_collapse_ws(p) for p in paras]

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return [_collapse_ws(text)] if text.strip() else []

    chunks: List[str] = []
    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if line.endswith((".", "!", "?", '."', '."', '."')):
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    if len(chunks) > 1:
        return [_collapse_ws(p) for p in chunks if p.strip()]
    return [_collapse_ws(text)] if text.strip() else []


def _table_nums_in_para(para: str) -> set:
    nums: set = set()
    for m in TABLE_REF_RE.finditer(para):
        body = _REF_KEYWORD_RE.sub("", m.group(0))
        for g in _REF_NUM_RE.findall(body):
            nums.add(normalize_table_number(g))
    return nums


def _mention_key(page: int, text: str) -> Tuple[int, str]:
    return page, text


def find_page_captions(page_text: str) -> List[dict]:
    captions = []
    for m in CAPTION_RE.finditer(page_text):
        start = m.start()
        rest = page_text[start:]
        para = re.split(r"\n\s*\n", rest, maxsplit=1)[0]
        caption = _collapse_ws(para)
        if not caption:
            continue
        captions.append(
            {
                "table_num": normalize_table_number(m.group(1)),
                "caption": caption,
                "start": start,
                "end": start + len(para),
            }
        )
    return captions


def pair_captions_to_tables(page_text: str) -> List[dict]:
    blocks = list(TABLE_BLOCK_RE.finditer(page_text))
    captions = find_page_captions(page_text)
    if not blocks:
        return []

    results: List[dict] = []
    if len(captions) == len(blocks):
        caps_sorted = sorted(captions, key=lambda c: c["start"])
        for bm, cap in zip(blocks, caps_sorted):
            results.append(
                {
                    "match": bm,
                    "caption": cap["caption"],
                    "table_num": cap["table_num"],
                }
            )
        return results

    used: set = set()
    for bm in blocks:
        t_start, t_end = bm.start(), bm.end()
        best_idx = None
        best_dist = float("inf")
        for ci, cap in enumerate(captions):
            if ci in used:
                continue
            if cap["end"] <= t_start:
                dist = t_start - cap["end"]
            elif cap["start"] >= t_end:
                dist = cap["start"] - t_end
            else:
                dist = 0
            if dist < best_dist:
                best_dist = dist
                best_idx = ci
        if best_idx is not None:
            used.add(best_idx)
            cap = captions[best_idx]
            results.append(
                {
                    "match": bm,
                    "caption": cap["caption"],
                    "table_num": cap["table_num"],
                }
            )
        else:
            results.append({"match": bm, "caption": "", "table_num": None})
    return results


def page_paragraphs_without_tables(page_text: str) -> List[str]:
    text = TABLE_BLOCK_RE.sub(" ", page_text)
    return _split_paragraphs(text)


_FOOTNOTE_URL_RE = re.compile(r"https?://|github\.com|\$\^\{\d+\}")


def _is_safe_continuation(para: str) -> bool:
    t = para.strip()
    if not t:
        return False
    if CAPTION_RE.match(t):
        return False
    if re.match(r"^Table\b", t, re.IGNORECASE):
        return False
    return True


_HYPHEN_CONTINUATION_PREFIX_RE = re.compile(r"^([a-z][a-z'-]{0,30})([.,;:!?])?")
_SENTENCE_END_RE = re.compile(r"""[.!?]['")\]]*\s*$""")
MAX_MENTION_EXTENSIONS = 2


def _ends_complete_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.rstrip()))


def _continuation_paragraphs(
    page_idx: int,
    para_idx: int,
    paras: List[str],
    pages: List[str],
):
    for k in range(para_idx + 1, len(paras)):
        yield paras[k]
    if page_idx < len(pages):
        for p in page_paragraphs_without_tables(pages[page_idx]):
            yield p


def _extract_hyphen_continuation_prefix(para: str) -> Optional[str]:
    """Leading lowercase word fragment after a hyphenated line break."""
    t = para.lstrip()
    if not t or not t[0].islower():
        return None
    if CAPTION_RE.match(t) or re.match(r"^Table\b", t, re.IGNORECASE):
        return None
    m = _HYPHEN_CONTINUATION_PREFIX_RE.match(t)
    if not m:
        return None
    return m.group(1) + (m.group(2) or "")


def _extend_mention_text(
    para: str,
    page_idx: int,
    para_idx: int,
    paras: List[str],
    pages: List[str],
) -> str:
    """Rejoin hyphen breaks and incomplete sentences across paragraphs or pages."""
    out = para.strip()
    extensions = 0

    for nxt in _continuation_paragraphs(page_idx, para_idx, paras, pages):
        if extensions >= MAX_MENTION_EXTENSIONS:
            break

        nxt_text = nxt.strip()
        if not nxt_text:
            continue

        if out.rstrip().endswith("-"):
            merged_base = out.rstrip()[:-1]
            if _is_safe_continuation(nxt_text):
                out = _collapse_ws(merged_base + nxt_text)
                extensions += 1
                continue
            prefix = _extract_hyphen_continuation_prefix(nxt_text)
            if prefix:
                out = _collapse_ws(merged_base + prefix)
                break
            continue

        if _ends_complete_sentence(out) or out.rstrip().endswith(":"):
            break

        if not _is_safe_continuation(nxt_text):
            continue

        if extensions == 0 and not nxt_text[0].islower():
            break

        out = _collapse_ws(out + " " + nxt_text)
        extensions += 1

    return out


def _is_usable_mention(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_MENTION_CHARS:
        return False
    if re.fullmatch(r"-+", t):
        return False
    if _HEADING_ONLY_RE.match(t) and len(t) < 80:
        return False
    if t.startswith("$$") or t.startswith("\\["):
        return False
    url_hits = len(_FOOTNOTE_URL_RE.findall(t))
    if url_hits >= 2:
        return False
    if url_hits >= 1 and len(t) < 200:
        return False
    return True


def find_mentions(pages: List[str]) -> Dict[str, List[dict]]:
    """Map table number -> narrative paragraphs that cite it (+ nearby paragraphs)."""
    mentions: Dict[str, List[dict]] = {}
    seen: Dict[str, set] = {}

    for page_idx, page_text in enumerate(pages, start=1):
        paras = page_paragraphs_without_tables(page_text)
        for i, para in enumerate(paras):
            if CAPTION_RE.match(para):
                continue
            nums = _table_nums_in_para(para)
            if not nums:
                continue

            lo = max(0, i - ADJACENT_PARA_WINDOW)
            hi = min(len(paras), i + ADJACENT_PARA_WINDOW + 1)
            for j in range(lo, hi):
                candidate = paras[j]
                if CAPTION_RE.match(candidate):
                    continue
                if j != i:
                    adj_nums = _table_nums_in_para(candidate)
                    if adj_nums and adj_nums.isdisjoint(nums):
                        continue
                text = _extend_mention_text(candidate, page_idx, j, paras, pages)
                if not _is_usable_mention(text):
                    continue
                for n in nums:
                    key = _mention_key(page_idx, text)
                    seen.setdefault(n, set())
                    if key in seen[n]:
                        continue
                    seen[n].add(key)
                    mentions.setdefault(n, []).append({"page": page_idx, "text": text})

    for n in mentions:
        mentions[n].sort(key=lambda m: (m["page"], m["text"]))
    return mentions


# ---------------------------------------------------------------------------
# Table HTML
# ---------------------------------------------------------------------------


def header_and_rows_from_html(html: str) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    thead = soup.find("thead")
    tbody = soup.find("tbody")

    def _row_text(tr):
        cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        return " | ".join(cells) if cells else None

    header_lines: List[str] = []
    body_lines: List[str] = []

    if thead is not None:
        for tr in thead.find_all("tr"):
            txt = _row_text(tr)
            if txt:
                header_lines.append(txt)

    trs = tbody.find_all("tr") if tbody is not None else soup.find_all("tr")
    for tr in trs:
        if thead is not None and tr in thead.find_all("tr"):
            continue
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        txt = _row_text(tr)
        if not txt:
            continue
        if all(c.name == "th" for c in cells) and not body_lines:
            header_lines.append(txt)
        else:
            body_lines.append(txt)

    if not header_lines and body_lines:
        header_lines = [body_lines[0]]
        body_lines = body_lines[1:]
    return header_lines, body_lines


# ---------------------------------------------------------------------------
# Normalization + GLiNER2
# ---------------------------------------------------------------------------


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = _strip_accents(s)
    return re.sub(r"\s+", " ", s)


def normalize_dataset(text: object) -> str:
    s = normalize_text(text)
    return s.replace(" ", "").replace("_", "").replace("-", "")


def _strip_trailing_plural(s: str) -> str:
    if not s or "@" in s or not s.isalpha():
        return s
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def normalize_metric(text: object) -> str:
    s = normalize_text(text)
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9@]+", "", s)
    return _strip_trailing_plural(s)


def metric_dedup_key(raw: str) -> str:
    nm = normalize_metric(raw)
    if not nm:
        return ""
    m = re.match(r"^h@(\d+)$", nm)
    if m:
        return f"hits@{m.group(1)}"
    return nm


def _is_incomplete_metric_key(key: str) -> bool:
    """Normalized metric keys ending in @ without a numeric suffix (table header fragments)."""
    if not key or "@" not in key:
        return False
    suffix = key.split("@", 1)[1]
    return not suffix or not any(ch.isdigit() for ch in suffix)


def _pick_display(candidates: List[str]) -> str:
    return max(candidates, key=lambda s: (len(s), any(c.isupper() for c in s)))


def _dedup_by_key(items: List[str], key_fn) -> List[str]:
    buckets: Dict[str, List[str]] = {}
    for value in items:
        key = key_fn(value)
        if not key:
            continue
        buckets.setdefault(key, []).append(value)
    return sorted(_pick_display(vals) for vals in buckets.values())


def _clean_entity(value: str) -> str:
    s = str(value).strip()
    s = re.sub(r"\[[^\]]{1,50}\]", "", s)
    s = re.sub(
        r"\((?:[^\)]*\d{4}[^\)]*|[^\)]*et\s*al\.?[^\)]*)\)",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\$([^$]+?)\$", r"\1", s)
    s = re.sub(r"\\(?:textbf|textit|text|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\1", s)
    for _ in range(3):
        s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip(" .,:;-")
    return s


def _extract_entities_raw(text: str) -> Dict[str, Tuple[str, float]]:
    best: Dict[str, Tuple[str, float]] = {}
    if not text or not text.strip():
        return best
    try:
        result = gliner2_model.extract_entities(
            text[:GLINER2_MAX_CHARS],
            GLINER2_LABEL_DESCRIPTIONS,
            include_confidence=True,
        )
    except Exception as e:
        log.warning("GLiNER2 error: %s: %s", type(e).__name__, e)
        return best

    for label, items in (result or {}).get("entities", {}).items():
        if label not in GLINER2_LABELS:
            continue
        for item in items or []:
            if isinstance(item, dict):
                raw, score = str(item.get("text", "")), float(item.get("confidence", 1.0) or 1.0)
            else:
                raw, score = str(item), 1.0
            value = _clean_entity(raw)
            if score < GLINER2_MIN_SCORE or len(value) < 2:
                continue
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                continue
            prev = best.get(value)
            if prev is None or score > prev[1]:
                best[value] = (label, score)
    return best


def _merge_best(target: Dict[str, Tuple[str, float]], other: Dict[str, Tuple[str, float]]) -> None:
    for value, (label, score) in other.items():
        prev = target.get(value)
        if prev is None or score > prev[1]:
            target[value] = (label, score)


def _keys_from_best(best: Dict[str, Tuple[str, float]], label: str, key_fn) -> set:
    keys = set()
    for value, (lbl, _) in best.items():
        if lbl != label:
            continue
        k = key_fn(value)
        if k:
            keys.add(k)
    return keys


def extract_table_entities(caption: str, html: str) -> Dict[str, List[str]]:
    header_lines, body_lines = header_and_rows_from_html(html)
    header_context = "\n".join(header_lines)

    table_best: Dict[str, Tuple[str, float]] = {}

    if header_context:
        _merge_best(table_best, _extract_entities_raw(header_context))

    for row_text in body_lines:
        prompt = (
            f"Table column headers: {header_context}\nRow: {row_text}"
            if header_context
            else row_text
        )
        _merge_best(table_best, _extract_entities_raw(prompt))

    if not table_best:
        full = "\n".join(header_lines + body_lines)
        _merge_best(table_best, _extract_entities_raw(full))

    caption_only_ds: set = set()
    caption_only_mt: set = set()
    if caption:
        cap_best = _extract_entities_raw(caption)
        cap_ds = _keys_from_best(cap_best, "dataset", normalize_dataset)
        cap_mt = _keys_from_best(cap_best, "metric", metric_dedup_key)
        table_ds = _keys_from_best(table_best, "dataset", normalize_dataset)
        table_mt = _keys_from_best(table_best, "metric", metric_dedup_key)
        caption_only_ds = cap_ds - table_ds
        caption_only_mt = cap_mt - table_mt

    filtered_datasets: List[str] = []
    filtered_metrics: List[str] = []
    for value, (label, _) in table_best.items():
        if label == "model":
            continue
        if label == "dataset":
            k = normalize_dataset(value)
            if k and k not in caption_only_ds:
                filtered_datasets.append(value)
        elif label == "metric":
            k = metric_dedup_key(value)
            if k and not _is_incomplete_metric_key(k) and k not in caption_only_mt:
                filtered_metrics.append(value)

    return {
        "dataset": _dedup_by_key(filtered_datasets, normalize_dataset),
        "metric": _dedup_by_key(filtered_metrics, metric_dedup_key),
    }


# ---------------------------------------------------------------------------
# LaTeX → plain text (pylatexenc)
# ---------------------------------------------------------------------------

_latex2text = None
_LATEX_MATH_RE = re.compile(
    r"\$\$([^$]+)\$\$|\$([^$]+)\$|\\\(([^)]+)\\\)|\\\[([^\]]+)\\\]"
)


def _get_latex2text():
    global _latex2text
    if _latex2text is None:
        from pylatexenc.latex2text import LatexNodes2Text

        _latex2text = LatexNodes2Text()
    return _latex2text


def _convert_latex_fragment(fragment: str) -> str:
    try:
        return _get_latex2text().latex_to_text(fragment)
    except Exception as e:
        log.debug("latex2text: %s", e)
        return fragment


def latex_to_plain(text: str) -> str:
    """Convert inline/display LaTeX segments to readable Unicode; leave plain text intact."""
    if not text or not str(text).strip():
        return text

    def _repl(match: re.Match) -> str:
        fragment = next(g for g in match.groups() if g is not None)
        return _convert_latex_fragment(fragment)

    return _LATEX_MATH_RE.sub(_repl, str(text))


def _format_mentions_for_export(mentions: List[dict]) -> List[dict]:
    return [{"page": m["page"], "text": latex_to_plain(m["text"])} for m in mentions]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def extract_context_for_pdf(
    pdf_path: Path,
    *,
    ocr_cache_dir: Path,
    force_ocr: bool = False,
) -> dict:
    log.info("Processing: %s", pdf_path.name)
    pages = ocr_pdf_pages(pdf_path, ocr_cache_dir, force_ocr=force_ocr)
    mentions_by_num = find_mentions(pages)

    tables = []
    for page_idx, page_text in enumerate(pages, start=1):
        paired = pair_captions_to_tables(page_text)
        for t_i, item in enumerate(paired, start=1):
            html = item["match"].group(0)
            caption_raw = item["caption"] or ""
            table_num = item["table_num"]
            mentions_raw = mentions_by_num.get(table_num, []) if table_num else []
            ents = extract_table_entities(caption_raw, html)

            tables.append(
                {
                    "table_id": f"{pdf_path.stem}_p{page_idx}_t{t_i}",
                    "table_label": f"Table {table_num}" if table_num else None,
                    "page": page_idx,
                    "caption": latex_to_plain(caption_raw),
                    "mentions": _format_mentions_for_export(mentions_raw),
                    "datasets": ents["dataset"],
                    "metrics": ents["metric"],
                }
            )

    log.info(
        "  tables=%d | with_caption=%d | datasets=%d | metrics=%d",
        len(tables),
        sum(1 for t in tables if t["caption"]),
        sum(len(t["datasets"]) for t in tables),
        sum(len(t["metrics"]) for t in tables),
    )

    return {
        "paper": pdf_path.stem,
        "source_pdf": str(pdf_path.resolve()),
        "num_tables": len(tables),
        "tables": tables,
    }


def collect_pdf_paths(input_path: Path) -> List[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files in {input_path}")
        return pdfs
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_extract(
    input_path: Path,
    output_dir: Path,
    *,
    ocr_cache_dir: Optional[Path] = None,
    force_ocr: bool = False,
    skip_existing: bool = False,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ocr_cache_dir or (output_dir / "ocr_cache")

    load_models()
    pdf_paths = collect_pdf_paths(input_path)
    written: List[Path] = []

    for i, pdf_path in enumerate(pdf_paths, start=1):
        out_file = output_dir / f"{pdf_path.stem}.json"
        if skip_existing and out_file.exists():
            log.info("[%d/%d] skip existing: %s", i, len(pdf_paths), out_file.name)
            written.append(out_file)
            continue

        log.info("[%d/%d] %s", i, len(pdf_paths), pdf_path.name)
        doc = extract_context_for_pdf(
            pdf_path,
            ocr_cache_dir=cache_dir,
            force_ocr=force_ocr,
        )
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        log.info("  saved: %s", out_file)
        written.append(out_file)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_table_context_lightonocr",
        description=(
            "Extract per-table context (caption + mentions) and entities "
            "(datasets & metrics) from PDFs using LightOnOCR + GLiNER2."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="PDF file or directory containing *.pdf files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory; writes <pdf_stem>.json per PDF.",
    )
    parser.add_argument(
        "--ocr-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help="OCR page cache directory (default: <output>/ocr_cache).",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Ignore OCR cache and re-run LightOnOCR.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose output JSON already exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        written = run_extract(
            args.input,
            args.output,
            ocr_cache_dir=args.ocr_cache,
            force_ocr=args.force_ocr,
            skip_existing=args.skip_existing,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1

    log.info("Done. %d JSON file(s) in %s", len(written), args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
