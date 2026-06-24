import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "rag_chunks"
HTML_DIR = ROOT / "日文页面"
JSONL_OUT = OUT_DIR / "qa_chunks.jsonl"
MD_OUT = OUT_DIR / "qa_chunks.md"

BASE_URL = "https://shadowverse-evolve.com"
QUESTION_URL = f"{BASE_URL}/question/"
BASIC_FAQ_URL = f"{BASE_URL}/question/faq/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    )
}


@dataclass(frozen=True)
class Expansion:
    ex: str
    title: str | None
    url: str


@dataclass(frozen=True)
class CardLink:
    card_no: str
    name: str | None
    url: str
    image_url: str | None


class Fetcher:
    def __init__(
        self,
        delay: float,
        timeout: float,
        retries: int,
        cache_html: bool,
        use_cache: bool,
    ):
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.cache_html = cache_html
        self.use_cache = use_cache
        HTML_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, cache_name: str | None = None) -> str:
        cache_path = HTML_DIR / cache_name if cache_name else None
        if self.use_cache and cache_path and cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path.read_text(encoding="utf-8")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, headers=HEADERS, timeout=self.timeout)
                response.raise_for_status()
                response.encoding = "utf-8"
                text = response.text
                if self.cache_html and cache_path:
                    cache_path.write_text(text, encoding="utf-8")
                if self.delay:
                    time.sleep(self.delay)
                return text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(max(self.delay, 1.0) * (attempt + 1))

        raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def clean_text(node) -> str | None:
    if node is None:
        return None
    text = node.get_text("", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def strip_label(text: str | None, label: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith(label):
        text = text[len(label) :].strip()
    return text or None


def qa_id(source_type: str, qno: str | None, card_no: str | None, index: int) -> str:
    parts = ["qa", source_type]
    if card_no:
        parts.append(card_no)
    if qno:
        parts.append(f"Q{qno}")
    else:
        parts.append(str(index))
    safe = "_".join(parts)
    return re.sub(r"[^0-9A-Za-z]+", "_", safe).strip("_")


def parse_qa_title(title: str | None) -> tuple[str | None, str | None]:
    if not title:
        return None, None
    match = re.search(r"Q\s*(\d+)(?:[\(\uff08]([0-9]{4}-[0-9]{2}-[0-9]{2})[\)\uff09])?", title)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def normalize_card_url(href: str) -> tuple[str, str | None]:
    absolute = urljoin(BASE_URL, href.replace("&amp;", "&"))
    parsed = urlparse(absolute)
    params = parse_qs(parsed.query)
    card_no = (params.get("cardno") or [None])[0]
    if not card_no:
        return absolute, None
    query = urlencode({"cardno": card_no})
    return f"{BASE_URL}/cardlist/?{query}&faq", card_no


def discover_expansions(html: str) -> list[Expansion]:
    soup = soup_from_html(html)
    expansions: dict[str, Expansion] = {}
    for anchor in soup.select('a[href*="/question/card?ex="], a[href*="question/card?ex="]'):
        href = anchor.get("href")
        if not href:
            continue
        absolute = urljoin(BASE_URL, href.replace("&amp;", "&"))
        ex = (parse_qs(urlparse(absolute).query).get("ex") or [None])[0]
        if not ex:
            continue
        title = clean_text(anchor)
        if not title:
            img = anchor.select_one("img")
            title = img.get("alt") if img else None
        expansions.setdefault(ex, Expansion(ex=ex, title=title, url=absolute))
    return sorted(expansions.values(), key=lambda item: item.ex)


def parse_expansion_cards(html: str) -> list[CardLink]:
    soup = soup_from_html(html)
    cards: dict[str, CardLink] = {}
    for anchor in soup.select('ul.cardlist-Result_List_Gallery a[href*="cardno="]'):
        href = anchor.get("href")
        if not href:
            continue
        url, card_no = normalize_card_url(href)
        if not card_no:
            continue
        img = anchor.select_one("img")
        name = anchor.get("title") or (img.get("alt") if img else None) or clean_text(anchor)
        image_url = urljoin(BASE_URL, img.get("src")) if img and img.get("src") else None
        cards.setdefault(card_no, CardLink(card_no=card_no, name=name, url=url, image_url=image_url))
    return list(cards.values())


def parse_card_detail(html: str, card: CardLink, expansion: Expansion | None) -> list[dict]:
    soup = soup_from_html(html)
    card_name = clean_text(soup.select_one(".cardlist-Detail h1.ttl")) or card.name
    product = clean_text(soup.select_one(".cardlist-Detail_Products .ttl"))
    source_url = card.url
    chunks = []
    for index, item in enumerate(soup.select(".qa-List_Item"), start=1):
        title = clean_text(item.select_one(".qa-List_Ttl"))
        qno, date = parse_qa_title(title)
        question = strip_label(clean_text(item.select_one(".qa-List_Txt-Q")), "Q")
        answer = strip_label(clean_text(item.select_one(".qa-List_Txt-A")), "A")
        if not question and not answer:
            continue
        chunk = {
            "id": qa_id("card", qno, card.card_no, index),
            "source_type": "qa_card",
            "qa_no": qno,
            "date": date,
            "question": question,
            "answer": answer,
            "card_no": card.card_no,
            "card_name_jp": card_name,
            "expansion": expansion.ex if expansion else None,
            "expansion_title": expansion.title if expansion else product,
            "product": product,
            "source_url": source_url,
            "image_url": card.image_url,
        }
        chunk["embedding_text"] = build_embedding_text(chunk)
        chunks.append(chunk)
    return chunks


def nearest_heading_text(item) -> str | None:
    for previous in item.find_all_previous(["h2", "h3", "h4"]):
        text = clean_text(previous)
        if text and text != "Q&A":
            return text
    return None


def parse_basic_faq(html: str) -> list[dict]:
    soup = soup_from_html(html)
    chunks = []
    for index, item in enumerate(soup.select(".qa-List_Item"), start=1):
        title = clean_text(item.select_one(".qa-List_Ttl"))
        qno, date = parse_qa_title(title)
        question = strip_label(clean_text(item.select_one(".qa-List_Txt-Q")), "Q")
        answer = strip_label(clean_text(item.select_one(".qa-List_Txt-A")), "A")
        if not question and not answer:
            continue
        chunk = {
            "id": qa_id("basic", qno, None, index),
            "source_type": "qa_basic",
            "qa_no": qno,
            "date": date,
            "question": question,
            "answer": answer,
            "category": nearest_heading_text(item),
            "source_url": BASIC_FAQ_URL,
        }
        chunk["embedding_text"] = build_embedding_text(chunk)
        chunks.append(chunk)
    return chunks


def build_embedding_text(chunk: dict) -> str:
    labels = [
        ("QA番号", chunk.get("qa_no")),
        ("日付", chunk.get("date")),
        ("種別", chunk.get("source_type")),
        ("カード番号", chunk.get("card_no")),
        ("カード名", chunk.get("card_name_jp")),
        ("商品コード", chunk.get("expansion")),
        ("商品名", chunk.get("expansion_title") or chunk.get("product")),
        ("カテゴリ", chunk.get("category")),
        ("質問", chunk.get("question")),
        ("回答", chunk.get("answer")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in labels if value)


def write_jsonl(chunks: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_markdown(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Shadowverse EVOLVE Q&A chunks", ""]
    for chunk in chunks:
        title_parts = []
        if chunk.get("qa_no"):
            title_parts.append(f"Q{chunk['qa_no']}")
        if chunk.get("card_no"):
            title_parts.append(chunk["card_no"])
        if chunk.get("card_name_jp"):
            title_parts.append(chunk["card_name_jp"])
        heading = " / ".join(title_parts) or chunk["id"]
        lines.extend(
            [
                f"## {heading}",
                "",
                f"- id: `{chunk['id']}`",
                f"- source: {chunk['source_url']}",
            ]
        )
        if chunk.get("date"):
            lines.append(f"- date: {chunk['date']}")
        if chunk.get("expansion"):
            lines.append(f"- expansion: {chunk['expansion']}")
        if chunk.get("category"):
            lines.append(f"- category: {chunk['category']}")
        lines.extend(
            [
                "",
                f"Q: {chunk.get('question') or ''}",
                "",
                f"A: {chunk.get('answer') or ''}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def sort_key(chunk: dict) -> tuple:
    source_rank = 0 if chunk.get("source_type") == "qa_basic" else 1
    qa_no = chunk.get("qa_no")
    try:
        qa_no_key = int(qa_no) if qa_no is not None else -1
    except ValueError:
        qa_no_key = -1
    return (
        source_rank,
        chunk.get("expansion") or "",
        chunk.get("card_no") or "",
        qa_no_key,
        chunk.get("id") or "",
    )


def run(args: argparse.Namespace) -> list[dict]:
    fetcher = Fetcher(
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
        cache_html=args.cache_html,
        use_cache=args.use_cache,
    )

    chunks: list[dict] = []

    if not args.skip_basic:
        basic_html = fetcher.get(BASIC_FAQ_URL, "QA_basic.html")
        basic_chunks = parse_basic_faq(basic_html)
        print(f"basic faq: {len(basic_chunks)} QA")
        chunks.extend(basic_chunks)

    index_html = fetcher.get(QUESTION_URL, "QA.html")
    expansions = discover_expansions(index_html)
    if args.expansion:
        wanted = set(args.expansion)
        expansions = [item for item in expansions if item.ex in wanted]
    if args.max_expansions:
        expansions = expansions[: args.max_expansions]
    print(f"expansions: {len(expansions)}")

    for expansion_index, expansion in enumerate(expansions, start=1):
        expansion_html = fetcher.get(expansion.url, f"QA_{expansion.ex}.html")
        cards = parse_expansion_cards(expansion_html)
        if args.max_cards_per_expansion:
            cards = cards[: args.max_cards_per_expansion]
        print(f"[{expansion_index}/{len(expansions)}] {expansion.ex}: {len(cards)} cards")

        def scrape_card(card: CardLink) -> tuple[CardLink, list[dict]]:
            cache_name = f"QA_{card.card_no.replace('-', '_')}.html"
            card_html = fetcher.get(card.url, cache_name)
            return card, parse_card_detail(card_html, card, expansion)

        if args.workers <= 1:
            for card_index, card in enumerate(cards, start=1):
                scraped_card, card_chunks = scrape_card(card)
                print(f"  [{card_index}/{len(cards)}] {scraped_card.card_no}: {len(card_chunks)} QA")
                chunks.extend(card_chunks)
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(scrape_card, card) for card in cards]
                for future in as_completed(futures):
                    scraped_card, card_chunks = future.result()
                    completed += 1
                    print(f"  [{completed}/{len(cards)}] {scraped_card.card_no}: {len(card_chunks)} QA")
                    chunks.extend(card_chunks)

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Shadowverse EVOLVE Q&A into RAG chunks.")
    parser.add_argument("--expansion", action="append", help="Only scrape one expansion code, e.g. BP01. Can repeat.")
    parser.add_argument("--skip-basic", action="store_true", help="Skip /question/faq/.")
    parser.add_argument("--cache-html", action="store_true", help="Save downloaded HTML under 日文页面.")
    parser.add_argument("--use-cache", action="store_true", help="Use non-empty cached HTML before downloading.")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between requests, in seconds.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout, in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for HTTP failures.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent card detail fetches per expansion.")
    parser.add_argument("--max-expansions", type=int, help="Debug limit for expansion count.")
    parser.add_argument("--max-cards-per-expansion", type=int, help="Debug limit for card count per expansion.")
    parser.add_argument("--jsonl-out", type=Path, default=JSONL_OUT)
    parser.add_argument("--md-out", type=Path, default=MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = sorted(run(args), key=sort_key)
    write_jsonl(chunks, args.jsonl_out)
    write_markdown(chunks, args.md_out)
    print(f"wrote {len(chunks)} chunks")
    print(f"jsonl: {args.jsonl_out}")
    print(f"markdown: {args.md_out}")


if __name__ == "__main__":
    main()
