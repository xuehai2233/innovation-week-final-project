import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sve_card_full_data.csv"
OUT_DIR = ROOT / "rag_chunks"
JSONL_OUT = OUT_DIR / "card_chunks.jsonl"
MD_OUT = OUT_DIR / "card_chunks_sample.md"


KEYWORD_SEEDS = [
    "入场曲",
    "谢幕曲",
    "进化",
    "进化时",
    "超进化",
    "超进化时",
    "攻击时",
    "守护",
    "疾驰",
    "突进",
    "指定攻击",
    "威压",
    "虹吸",
    "毁灭",
    "灵气",
    "快速",
    "蓄积",
    "土之秘术",
    "魔力连锁",
    "唤灵充能",
    "真红",
    "觉醒",
    "连击",
    "用餐",
    "参赛",
    "抽取",
    "抽卡",
    "舍弃",
    "丢弃",
    "寻找",
    "洗切",
    "伤害",
    "回复",
    "生命值",
    "攻击力",
    "消灭",
    "消失",
    "破坏",
    "横置",
    "竖置",
    "装备",
    "EX区域",
    "额外区",
    "墓场",
    "牌组",
    "手牌",
    "场上",
    "主战者",
    "从者",
    "随从",
    "护符",
    "法术",
]


def none_if_blank(value: str):
    value = (value or "").strip()
    return value if value else None


def int_or_none(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def card_id(card_no: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", card_no.strip()).strip("_")
    return f"card_{safe}"


def split_related(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[\s,;，；]+", value.strip()) if part]


def extract_keywords(row: dict) -> list[str]:
    text = "\n".join(
        [
            row.get("name_cn", ""),
            row.get("name_jp", ""),
            row.get("type", ""),
            row.get("title", ""),
            row.get("race", ""),
            row.get("desc_cn", ""),
            row.get("desc_jp", ""),
        ]
    )
    keywords = []

    for field in ["craft", "card_type", "type", "title", "race"]:
        value = (row.get(field) or "").strip()
        if value:
            for part in re.split(r"[・/／,，、\s]+", value):
                if part:
                    keywords.append(part)

    for marker in re.findall(r"[《【]([^》】]+)[》】]", text):
        marker = marker.strip()
        if marker:
            keywords.append(marker)

    for seed in KEYWORD_SEEDS:
        if seed in text:
            keywords.append(seed)

    # Common Japanese ability words help Japanese queries hit Chinese cards.
    jp_map = {
        "ファンファーレ": "ファンファーレ",
        "ラストワード": "ラストワード",
        "進化時": "進化時",
        "超進化時": "超進化時",
        "疾走": "疾走",
        "守護": "守護",
        "突進": "突進",
        "指定攻撃": "指定攻撃",
        "威圧": "威圧",
        "オーラ": "オーラ",
        "クイック": "クイック",
        "土の秘術": "土の秘術",
    }
    for seed, keyword in jp_map.items():
        if seed in text:
            keywords.append(keyword)

    seen = set()
    result = []
    for item in keywords:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result[:32]


def build_embedding_text(chunk: dict) -> str:
    related = " ".join(chunk["related_card_nos"]) if chunk["related_card_nos"] else "无"
    keywords = " ".join(chunk["keywords"]) if chunk["keywords"] else "无"
    fields = [
        ("卡号", chunk["card_no"]),
        ("中文名", chunk["name_cn"]),
        ("日文名", chunk["name_jp"]),
        ("职业", chunk["craft"]),
        ("卡牌种类", chunk["card_type"]),
        ("类型", chunk["type"]),
        ("费用", chunk["cost"]),
        ("攻击力", chunk["attack"]),
        ("生命值", chunk["life"]),
        ("稀有度", chunk["rare"]),
        ("出处", chunk["set"]),
        ("作品/标题", chunk["title"]),
        ("种族", chunk["race"]),
        ("中文效果", chunk["desc_cn"]),
        ("日文效果", chunk["desc_jp"]),
        ("关联卡", related),
        ("关键词", keywords),
    ]
    lines = []
    for label, value in fields:
        if value is None or value == "":
            continue
        lines.append(f"{label}：{value}")
    return "\n".join(lines)


def build_chunks() -> list[dict]:
    chunks = []
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_no = (row.get("card_no") or "").strip()
            if not card_no:
                continue

            chunk = {
                "id": card_id(card_no),
                "source_type": "card",
                "card_no": card_no,
                "name_cn": none_if_blank(row.get("name_cn", "")),
                "name_jp": none_if_blank(row.get("name_jp", "")),
                "craft": none_if_blank(row.get("craft", "")),
                "card_type": none_if_blank(row.get("card_type", "")),
                "cost": int_or_none(row.get("cost", "")),
                "attack": int_or_none(row.get("attack", "")),
                "life": int_or_none(row.get("life", "")),
                "type": none_if_blank(row.get("type", "")),
                "desc_cn": none_if_blank(row.get("desc_cn", "")),
                "desc_jp": none_if_blank(row.get("desc_jp", "")),
                "related_card_nos": split_related(row.get("related_card_nos", "")),
                "keywords": extract_keywords(row),
                "rare": none_if_blank(row.get("rare", "")),
                "set": none_if_blank(row.get("from", "")),
                "title": none_if_blank(row.get("title", "")),
                "race": none_if_blank(row.get("race", "")),
                "drawer": none_if_blank(row.get("drawer", "")),
                "speech": none_if_blank(row.get("speech", "")),
                "source": "sve_card_full_data.csv",
            }
            chunk["embedding_text"] = build_embedding_text(chunk)
            chunks.append(chunk)
    return chunks


def write_outputs(chunks: list[dict]):
    OUT_DIR.mkdir(exist_ok=True)
    with JSONL_OUT.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with MD_OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Card Chunk Samples\n\n")
        for chunk in chunks[:20]:
            f.write("[card_chunk]\n")
            f.write(f"id: {chunk['id']}\n")
            f.write(f"card_no: {chunk['card_no']}\n")
            f.write(f"name_cn: {chunk['name_cn']}\n")
            f.write(f"name_jp: {chunk['name_jp']}\n")
            f.write(f"keywords: {', '.join(chunk['keywords'])}\n\n")
            f.write("embedding_text:\n")
            f.write(chunk["embedding_text"] + "\n\n")


def main():
    chunks = build_chunks()
    write_outputs(chunks)
    print(f"wrote {len(chunks)} card chunks")
    print(JSONL_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()
