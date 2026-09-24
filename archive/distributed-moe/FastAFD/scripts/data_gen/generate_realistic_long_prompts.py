#!/usr/bin/env python3
"""Generate long prompts by slicing real public-domain/open text corpora.

Each output line is one contiguous chunk from one source document. The script
does not add case headers, domain labels, instructions, tables, or synthetic
padding to the prompt text. Sources are downloaded into a cache and then sliced
with the target tokenizer so every line has the requested token length.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from minisgl.hf_support import load_tokenizer


@dataclass(frozen=True)
class Source:
    name: str
    domain: str
    urls: tuple[str, ...]


def pg_source(name: str, domain: str, ebook_id: int) -> Source:
    eid = str(ebook_id)
    return Source(
        name=name,
        domain=domain,
        urls=(
            f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt",
            f"https://www.gutenberg.org/files/{eid}/{eid}-0.txt",
            f"https://www.gutenberg.org/files/{eid}/{eid}.txt",
            f"https://www.gutenberg.org/ebooks/{eid}.txt.utf-8",
        ),
    )


def rfc_source(name: str, domain: str, rfc: int) -> Source:
    return Source(
        name=name,
        domain=domain,
        urls=(f"https://www.rfc-editor.org/rfc/rfc{int(rfc)}.txt",),
    )


SOURCES: tuple[Source, ...] = (
    pg_source("Moby-Dick", "literature", 2701),
    pg_source("War and Peace", "literature", 2600),
    pg_source("Pride and Prejudice", "literature", 1342),
    pg_source("The Adventures of Sherlock Holmes", "literature", 1661),
    pg_source("Great Expectations", "literature", 1400),
    pg_source("Middlemarch", "literature", 145),
    pg_source("Jane Eyre", "literature", 1260),
    pg_source("Dracula", "literature", 345),
    pg_source("Frankenstein", "literature", 84),
    pg_source("The Count of Monte Cristo", "literature", 1184),
    pg_source("Crime and Punishment", "literature", 2554),
    pg_source("The Brothers Karamazov", "literature", 28054),
    pg_source("The Odyssey", "classics", 1727),
    pg_source("The Iliad", "classics", 6130),
    pg_source("The Republic", "philosophy", 1497),
    pg_source("Leviathan", "political_philosophy", 3207),
    pg_source("The Prince", "political_philosophy", 1232),
    pg_source("Democracy in America", "politics_history", 815),
    pg_source("The Federalist Papers", "politics_law", 1404),
    pg_source("Second Treatise of Government", "politics_law", 7370),
    pg_source("The Wealth of Nations", "economics", 3300),
    pg_source("On Liberty", "political_philosophy", 34901),
    pg_source("The Communist Manifesto", "politics_economics", 61),
    pg_source("On the Origin of Species", "biology", 1228),
    pg_source("The Voyage of the Beagle", "biology_travel", 944),
    pg_source("The Descent of Man", "biology", 2300),
    pg_source("Relativity: The Special and General Theory", "physics", 5001),
    pg_source("The Notebooks of Leonardo Da Vinci", "art_science", 5000),
    pg_source("The Art of War", "strategy", 132),
    pg_source("The History of the Peloponnesian War", "history", 7142),
    pg_source("The Education of Henry Adams", "history_memoir", 2044),
    pg_source("The Souls of Black Folk", "sociology_history", 408),
    pg_source("Up from Slavery", "memoir_education", 2376),
    pg_source("Walden", "nature_essay", 205),
    pg_source("The Jungle", "labor_public_health", 140),
    pg_source("A Treatise of Human Nature", "philosophy", 4705),
    rfc_source("HTTP Semantics", "internet_protocols", 9110),
    rfc_source("HTTP/2", "internet_protocols", 9113),
    rfc_source("TLS 1.3", "internet_security", 8446),
    rfc_source("QUIC Transport", "internet_protocols", 9000),
    rfc_source("TCP", "internet_protocols", 9293),
    rfc_source("IPv6", "internet_protocols", 8200),
    rfc_source("DNS Concepts", "internet_protocols", 1034),
    rfc_source("SMTP", "internet_protocols", 5321),
)


GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I | re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/real_corpus_sources"))
    parser.add_argument("--min-source-tokens", type=int, default=12000)
    parser.add_argument("--preview-chars", type=int, default=220)
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end() :]
    text = GUTENBERG_END.sub("", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [
        p
        for p in paragraphs
        if len(p) > 80
        and "PROJECT GUTENBERG" not in p.upper()
        and "END OF THE PROJECT" not in p.upper()
    ]
    return " ".join(" ".join(p.split()) for p in paragraphs)


def cache_name(source: Source, url: str) -> str:
    key = hashlib.sha1(f"{source.name}|{url}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.name.lower()).strip("_")
    return f"{source.domain}__{safe}__{key}.txt"


def fetch_source(source: Source, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for url in source.urls:
        path = cache_dir / cache_name(source, url)
        if path.exists() and path.stat().st_size > 4096:
            return path.read_text(encoding="utf-8", errors="ignore")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "mini-sglang-calib long-context prompt generator (research; contact: local)"
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="ignore")
            if len(text) > 4096:
                path.write_text(text, encoding="utf-8")
                time.sleep(0.2)
                return text
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    print(f"warning: failed to fetch {source.name}", file=sys.stderr)
    return None


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def exact_chunk(tokenizer, ids: list[int], start: int, target: int) -> str | None:
    max_len = min(len(ids) - start, target + 24)
    if max_len <= 0:
        return None
    lo = max(1, target - 48)
    hi = max_len
    best: str | None = None
    best_n = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        text = " ".join(tokenizer.decode(ids[start : start + mid], skip_special_tokens=False).split())
        n = token_len(tokenizer, text)
        if n <= target:
            best = text
            best_n = n
            lo = mid + 1
        else:
            hi = mid - 1
    if best_n == target:
        return best
    for length in range(max(1, target - 64), max_len + 1):
        text = " ".join(tokenizer.decode(ids[start : start + length], skip_special_tokens=False).split())
        if token_len(tokenizer, text) == target:
            return text
    return None


def build_corpus(tokenizer, cache_dir: Path, min_source_tokens: int) -> list[tuple[Source, list[int]]]:
    corpus: list[tuple[Source, list[int]]] = []
    for source in SOURCES:
        raw = fetch_source(source, cache_dir)
        if not raw:
            continue
        text = clean_text(raw)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= min_source_tokens:
            corpus.append((source, ids))
            print(f"source_ok domain={source.domain} name={source.name!r} tokens={len(ids)}")
        else:
            print(f"source_skip_short domain={source.domain} name={source.name!r} tokens={len(ids)}")
    if not corpus:
        raise RuntimeError("no usable corpus sources")
    return corpus


def sample_prompts(tokenizer, corpus: list[tuple[Source, list[int]]], count: int, target: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []
    seen_texts: set[str] = set()
    seen_windows: set[tuple[str, int]] = set()
    source_order = list(range(len(corpus)))
    attempts = 0
    while len(prompts) < count:
        attempts += 1
        if attempts > count * 800:
            raise RuntimeError(f"unable to sample {count} unique exact chunks; got {len(prompts)}")
        src_idx = source_order[len(prompts) % len(source_order)]
        if attempts % len(source_order) == 0:
            rng.shuffle(source_order)
        source, ids = corpus[src_idx]
        if len(ids) <= target + 64:
            continue
        max_start = len(ids) - target - 32
        start = rng.randint(0, max_start)
        # Snap backward a little so many chunks begin near punctuation or a paragraph-like boundary.
        for probe in range(start, max(0, start - 160), -1):
            piece = tokenizer.decode(ids[probe : probe + 4], skip_special_tokens=False)
            if piece.startswith((" ", "\n")) or piece[:1] in (".", "!", "?", ";", ":"):
                start = min(probe + 1, max_start)
                break
        window_key = (source.name, start // 512)
        if window_key in seen_windows:
            continue
        text = exact_chunk(tokenizer, ids, start, target)
        if not text or text in seen_texts:
            continue
        if re.search(r"Case packet|Domain:|\[DOC|row=\d+|sql_\d+", text):
            raise RuntimeError("synthetic marker leaked into corpus prompt")
        seen_windows.add(window_key)
        seen_texts.add(text)
        prompts.append(text)
    return prompts


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    corpus = build_corpus(tokenizer, args.cache_dir, args.min_source_tokens)
    prompts = sample_prompts(
        tokenizer,
        corpus,
        count=int(args.count),
        target=int(args.target_tokens),
        seed=int(args.seed),
    )
    lengths = [token_len(tokenizer, prompt) for prompt in prompts]
    if sorted(set(lengths)) != [int(args.target_tokens)]:
        raise RuntimeError(f"unexpected prompt token lengths: {sorted(set(lengths))}")
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("duplicate prompts generated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"prompt_tokens_add_special={args.target_tokens}")
    print(f"num_prompts={len(prompts)}")
    print(f"unique_prompts={len(set(prompts))}")
    print(f"sources_used={len(corpus)}")
    print(f"source_domains={len(set(source.domain for source, _ids in corpus))}")
    print(f"chars_per_prompt_min={min(len(prompt) for prompt in prompts)}")
    print(f"chars_per_prompt_max={max(len(prompt) for prompt in prompts)}")
    if args.preview_chars > 0:
        print(f"preview={prompts[0][:args.preview_chars]}")


if __name__ == "__main__":
    main()
