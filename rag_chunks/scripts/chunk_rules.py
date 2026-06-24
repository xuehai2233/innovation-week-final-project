import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "guide.md"
OUT_DIR = ROOT / "rag_chunks"
SECTION_JSONL_OUT = OUT_DIR / "section_chunks.jsonl"
SECTION_MD_OUT = OUT_DIR / "section_chunks.md"
RULE_JSONL_OUT = OUT_DIR / "rule_chunks.jsonl"
RULE_MD_OUT = OUT_DIR / "rule_chunks.md"
# Kept for compatibility with the first generated version.
LEGACY_SECTION_JSONL_OUT = OUT_DIR / "rules_chunks.jsonl"
LEGACY_SECTION_MD_OUT = OUT_DIR / "rules_chunks.md"


CHAPTER_RE = re.compile(r"^##\s+(\d{1,2})\s+(.+?)\s*$")
SECTION_RE = re.compile(r"^###\s+(\d{1,2}\.\d{1,2})\s+(.+?)\s*$")
RULE_RE = re.compile(r"^\*{0,2}(\d+(?:\.\d+){2,})\b")
REF_RE = re.compile(r"\b\d+(?:\.\d+){1,}\b")

KEYWORD_SEEDS = [
    "主战者",
    "随从",
    "护符",
    "法术",
    "进化",
    "攻击",
    "伤害",
    "回复",
    "破坏",
    "消失",
    "抽牌",
    "洗切",
    "寻找",
    "战场",
    "手牌",
    "牌组",
    "墓场",
    "额外区",
    "计算领域",
    "检查时点",
    "规则处理",
    "自动能力",
    "启动能力",
    "永续能力",
    "持续效果",
    "置换效果",
    "费用",
    "能量点",
    "进化点",
    "回合玩家",
    "非回合玩家",
    "胜利",
    "败北",
    "平局",
    "投降",
    "守护",
    "疾驰",
    "突进",
    "指定攻击",
    "威压",
    "虹吸",
    "毁灭",
    "灵气",
    "入场曲",
    "谢幕曲",
    "蓄积",
    "土之秘术",
    "魔力连锁",
    "唤灵充能",
    "真红",
    "用餐",
    "参赛",
]


def normalize_line(line: str) -> str:
    line = line.replace("\uf06c", "- ")
    line = re.sub(r"\*\*(\d+(?:\.\d+)+.*?)\*\*", r"\1", line)
    return line.rstrip()


def rule_tuple(rule_number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in rule_number.split("."))


def has_section_prefix(rule_number: str, section_number: str) -> bool:
    return rule_number == section_number or rule_number.startswith(section_number + ".")


def split_embedded_numbered_rules(line: str) -> list[str]:
    leading = RULE_RE.match(line.strip())
    if not leading:
        return [line]

    leading_number = leading.group(1)
    leading_tuple = rule_tuple(leading_number)
    split_points = []
    for match in re.finditer(r"\d+(?:\.\d+){2,}", line):
        candidate = match.group(0)
        if match.start() == leading.start(1):
            continue
        candidate_tuple = rule_tuple(candidate)
        if candidate_tuple[:2] == leading_tuple[:2] and candidate_tuple > leading_tuple:
            split_points.append(match.start())

    if not split_points:
        return [line]

    starts = [0] + split_points
    ends = split_points + [len(line)]
    return [line[start:end].strip() for start, end in zip(starts, ends) if line[start:end].strip()]


def clean_content_lines(lines: list[str]) -> list[str]:
    cleaned = []
    pending_rule = None

    for raw in lines:
        split_lines = raw.splitlines() or [raw]
        for split_line in split_lines:
            line = split_line
            if line.startswith("## "):
                line = line.lstrip("# ").strip()

            for line_part in split_embedded_numbered_rules(line):
                if pending_rule:
                    if line_part.strip():
                        cleaned.append(f"{pending_rule} {line_part.strip()}")
                        pending_rule = None
                    continue

                if re.fullmatch(r"\d+(?:\.\d+){2,}", line_part.strip()):
                    pending_rule = line_part.strip()
                    continue

                if line_part.strip() or not cleaned or cleaned[-1].strip():
                    cleaned.append(line_part)

    if pending_rule:
        cleaned.append(pending_rule)

    return cleaned


def split_embedded_rule(section_number: str, title: str):
    marker = f"{section_number}.1"
    compact = title.replace(" ", "")
    compact_marker = marker.replace(" ", "")
    if compact_marker not in compact:
        return title.strip(), None

    idx = title.find(marker)
    if idx == -1:
        idx = title.replace(" ", "").find(compact_marker)
        return title.strip(), None

    clean_title = title[:idx].strip()
    embedded = f"{marker} {title[idx + len(marker):].strip()}".strip()
    return clean_title, embedded


def extract_keywords(title: str, chapter_title: str, content: str) -> list[str]:
    keywords = []
    for candidate in [title, chapter_title]:
        for part in re.split(r"[\s/：:（）()、，,]+", candidate):
            part = part.strip()
            if part and not re.fullmatch(r"\d+(?:\.\d+)*", part):
                keywords.append(part)

    haystack = f"{title}\n{content}"
    for seed in KEYWORD_SEEDS:
        if seed in haystack:
            keywords.append(seed)

    seen = set()
    result = []
    for item in keywords:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result[:16]


def flush_section_chunk(chunks, current):
    if not current:
        return

    content_lines = clean_content_lines(current["content_lines"])
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()

    content = "\n".join(content_lines)
    rule_ids = []
    seen_rule_ids = set()
    for line in content_lines:
        match = RULE_RE.match(line.strip())
        if (
            match
            and has_section_prefix(match.group(1), current["rule_number"])
            and match.group(1) not in seen_rule_ids
        ):
            seen_rule_ids.add(match.group(1))
            rule_ids.append(match.group(1))

    referenced_rules = sorted(set(REF_RE.findall(content)) - set(rule_ids))
    chunk = {
        "id": "rule_" + current["rule_number"].replace(".", "_"),
        "title": current["title"],
        "rule_number": current["rule_number"],
        "chapter": current["chapter"],
        "chapter_number": current["chapter_number"],
        "parent_chapter": current["chapter"],
        "keywords": extract_keywords(current["title"], current["chapter"], content),
        "rule_ids": rule_ids,
        "referenced_rules": referenced_rules,
        "source": "guide.md",
        "source_line_start": current["line_start"],
        "source_line_end": current["line_end"],
        "content": content,
    }
    chunks.append(chunk)


def build_chunks():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    chunks = []
    current_chapter_number = ""
    current_chapter = ""
    current = None

    for line_number, raw in enumerate(lines, start=1):
        line = normalize_line(raw)

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter_number = chapter_match.group(1)
            chapter_title = chapter_match.group(2).strip()
            numeric = int(chapter_number)
            current_numeric = int(current_chapter_number) if current_chapter_number else 0
            if 1 <= numeric <= 15 and numeric >= current_numeric:
                flush_section_chunk(chunks, current)
                current = None
                current_chapter_number = chapter_number
                current_chapter = f"{chapter_number} {chapter_title}"
                continue

        section_match = SECTION_RE.match(line)
        if section_match:
            section_number = section_match.group(1)
            title = section_match.group(2).strip()

            if title in {"至"} or title.startswith("所示"):
                if current:
                    current["content_lines"].append(line.lstrip("# ").strip())
                    current["line_end"] = line_number
                continue

            clean_title, embedded_rule = split_embedded_rule(section_number, title)
            flush_section_chunk(chunks, current)
            current = {
                "rule_number": section_number,
                "title": clean_title,
                "chapter_number": current_chapter_number,
                "chapter": current_chapter,
                "line_start": line_number,
                "line_end": line_number,
                "content_lines": [],
            }
            if embedded_rule:
                current["content_lines"].append(embedded_rule)
            continue

        if current:
            current["content_lines"].append(line)
            current["line_end"] = line_number

    flush_section_chunk(chunks, current)
    return chunks


def build_rule_chunks(section_chunks):
    rule_chunks = []

    for section in section_chunks:
        current = None

        def flush_rule():
            if not current:
                return
            content_lines = current["content_lines"]
            while content_lines and not content_lines[0].strip():
                content_lines.pop(0)
            while content_lines and not content_lines[-1].strip():
                content_lines.pop()
            content = "\n".join(content_lines)
            referenced_rules = sorted(set(REF_RE.findall(content)) - {current["rule_number"]})
            rule_chunks.append(
                {
                    "id": "rule_" + current["rule_number"].replace(".", "_"),
                    "title": section["title"],
                    "rule_number": current["rule_number"],
                    "section_number": section["rule_number"],
                    "section_title": section["title"],
                    "section_id": section["id"],
                    "chapter": section["chapter"],
                    "chapter_number": section["chapter_number"],
                    "parent_chapter": section["parent_chapter"],
                    "keywords": extract_keywords(section["title"], section["chapter"], content),
                    "referenced_rules": referenced_rules,
                    "source": section["source"],
                    "source_line_start": section["source_line_start"],
                    "source_line_end": section["source_line_end"],
                    "content": content,
                }
            )

        seen_rule_numbers = set()
        valid_rule_numbers = set(section["rule_ids"])

        for line in section["content"].splitlines():
            match = RULE_RE.match(line.strip())
            if (
                match
                and match.group(1) in valid_rule_numbers
                and match.group(1) not in seen_rule_numbers
            ):
                flush_rule()
                seen_rule_numbers.add(match.group(1))
                current = {
                    "rule_number": match.group(1),
                    "content_lines": [line.strip()],
                }
            elif current:
                if line.strip() or current["content_lines"][-1].strip():
                    current["content_lines"].append(line)

        flush_rule()

    return rule_chunks


def write_jsonl(path, chunks):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_section_md(path, chunks):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 《影之诗：进化对决》综合规则 Section Chunks\n\n")
        f.write("切片粒度：按二级规则小节（如 7.4、10.7、12.13）切分，保留其下所有子规则编号。用于 RAG 第一阶段召回。\n\n")
        for chunk in chunks:
            f.write("[section_chunk]\n")
            f.write(f"id: {chunk['id']}\n")
            f.write(f"title: {chunk['title']}\n")
            f.write(f"rule_number: {chunk['rule_number']}\n")
            f.write(f"chapter: {chunk['chapter']}\n")
            f.write(f"keywords: {', '.join(chunk['keywords'])}\n")
            f.write(f"source_lines: {chunk['source_line_start']}-{chunk['source_line_end']}\n\n")
            f.write("content:\n")
            f.write(chunk["content"].strip() + "\n\n")


def write_rule_md(path, chunks):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 《影之诗：进化对决》综合规则 Rule Chunks\n\n")
        f.write("切片粒度：按单条规则编号（如 10.6.2.3、8.4.3.1.2）切分。用于精确引用和裁判回答出处。\n\n")
        for chunk in chunks:
            f.write("[rule_chunk]\n")
            f.write(f"id: {chunk['id']}\n")
            f.write(f"title: {chunk['title']}\n")
            f.write(f"rule_number: {chunk['rule_number']}\n")
            f.write(f"section: {chunk['section_number']} {chunk['section_title']}\n")
            f.write(f"chapter: {chunk['chapter']}\n")
            f.write(f"keywords: {', '.join(chunk['keywords'])}\n")
            f.write(f"source_lines: {chunk['source_line_start']}-{chunk['source_line_end']}\n\n")
            f.write("content:\n")
            f.write(chunk["content"].strip() + "\n\n")


def write_outputs(section_chunks, rule_chunks):
    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(SECTION_JSONL_OUT, section_chunks)
    write_jsonl(RULE_JSONL_OUT, rule_chunks)
    write_jsonl(LEGACY_SECTION_JSONL_OUT, section_chunks)
    write_section_md(SECTION_MD_OUT, section_chunks)
    write_rule_md(RULE_MD_OUT, rule_chunks)
    write_section_md(LEGACY_SECTION_MD_OUT, section_chunks)


def main():
    section_chunks = build_chunks()
    rule_chunks = build_rule_chunks(section_chunks)
    write_outputs(section_chunks, rule_chunks)
    print(f"wrote {len(section_chunks)} section chunks")
    print(f"wrote {len(rule_chunks)} rule chunks")
    print(SECTION_JSONL_OUT)
    print(RULE_JSONL_OUT)


if __name__ == "__main__":
    main()
