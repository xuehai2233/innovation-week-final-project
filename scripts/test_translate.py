#!/usr/bin/env python3
"""
测试脚本 - 翻译前 10 个 QA 条目验证质量
"""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(r"D:\code\file\sveruler workbuddy\scripts")))

from translate_qa import (
    build_term_mapping, build_card_mapping,
    translate_batch, load_entries, save_entries,
    update_embedding,
    TERMS_FILE, CARDS_FILE, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)
from pathlib import Path
from openai import OpenAI

WORKSPACE = Path(r"D:\code\file\sveruler workbuddy")
INPUT_FILE = WORKSPACE / "rag_chunks" / "qa_chunks.jsonl"
TEST_OUTPUT = WORKSPACE / "rag_chunks" / "qa_test_cn.jsonl"

print("加载映射...")
term_map = build_term_mapping(TERMS_FILE)
card_map = build_card_mapping(CARDS_FILE)

print("加载 QA...")
entries = load_entries(INPUT_FILE)[:10]  # 前10条

print(f"测试条目数: {len(entries)}")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

print("开始测试翻译...\n")
translated = translate_batch(client, entries, term_map, card_map)

for e in translated:
    e["embedding_text"] = update_embedding(e)

save_entries(translated, TEST_OUTPUT)

print("\n" + "=" * 60)
print("测试结果预览:")
for e in translated:
    print(f"\n--- {e['id']} ---")
    print(f"原Q: {e['question'][:60]}...")
    print(f"中Q: {e['Q_cn'][:60]}...")
    print(f"原A: {e['answer'][:60]}...")
    print(f"中A: {e['A_cn'][:60]}...")

print(f"\n输出文件: {TEST_OUTPUT}")
