#!/usr/bin/env python3
"""
后处理脚本 - 修复翻译失败的条目
检查 qa_chunks_cn.jsonl 中 Q_cn 或 A_cn 为空的条目，
使用 DeepSeek 重新翻译。
"""
import json
import re
import time
from pathlib import Path
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-2e776b679daa4bf581324cf13c9393d6"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

WORKSPACE = Path(r"D:\code\file\sveruler workbuddy")
OUTPUT_FILE = WORKSPACE / "rag_chunks" / "qa_chunks_cn.jsonl"
TERMS_FILE = WORKSPACE / "canonical_terms.json"

def load_entries(filepath):
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def save_entries(entries, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    print("检查翻译质量...")
    entries = load_entries(OUTPUT_FILE)

    # 找出需要修复的条目
    fix_list = []
    for e in entries:
        q_cn = e.get("Q_cn", "")
        a_cn = e.get("A_cn", "")
        if not q_cn or q_cn.startswith("[翻译失败]") or q_cn.startswith("[解析遗漏]"):
            fix_list.append(e)
        elif not a_cn or a_cn.startswith("[翻译失败]") or a_cn.startswith("[解析遗漏]"):
            fix_list.append(e)

    print(f"总条目: {len(entries)}, 需修复: {len(fix_list)}")

    if not fix_list:
        print("所有条目翻译完整！")
        return

    # 加载术语表
    with open(TERMS_FILE, 'r', encoding='utf-8') as f:
        terms_data = json.load(f)

    # 构建简化术语表
    term_map = {}
    for _, ti in terms_data.items():
        jp = ti.get("jp", "")
        cn = ti.get("canonical_cn", "")
        if jp and cn:
            term_map[jp] = cn

    # 初始化客户端
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 逐条修复
    fixed = 0
    for i, e in enumerate(fix_list):
        print(f"修复 {i+1}/{len(fix_list)}: {e['id']}")
        for attempt in range(3):
            try:
                prompt = f"""翻译这个影之诗进化对决 QA 条目为中文：

日文问题：{e['question']}

日文答案：{e['answer']}

输出格式（严格）：
Q_cn: <中文问题>
A_cn: <中文答案>"""

                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": f"你是影之诗进化对决翻译专家。术语：{', '.join(f'{k}→{v}' for k,v in list(term_map.items())[:50])}"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )

                result = response.choices[0].message.content.strip()
                q_match = re.search(r'Q_cn\s*[:：]\s*(.+?)(?=\nA_cn|\Z)', result, re.DOTALL)
                a_match = re.search(r'A_cn\s*[:：]\s*(.+)', result, re.DOTALL)

                if q_match:
                    e["Q_cn"] = q_match.group(1).strip()
                if a_match:
                    e["A_cn"] = a_match.group(1).strip()

                # 更新 embedding
                if e.get("Q_cn") and e.get("A_cn"):
                    orig = e.get("embedding_text", "")
                    e["embedding_text"] = orig + f"\nQ_CN:\n{e['Q_cn']}\n\nA_CN:\n{e['A_cn']}"
                    fixed += 1

                time.sleep(1)
                break
            except Exception as ex:
                print(f"  重试 {attempt+1}: {ex}")
                time.sleep(5)

    # 保存
    save_entries(entries, OUTPUT_FILE)
    print(f"\n修复完成。成功修复: {fixed}/{len(fix_list)}")

    # 最终统计
    entries = load_entries(OUTPUT_FILE)
    fails = sum(1 for e in entries if not e.get("Q_cn") or e["Q_cn"].startswith("[翻译失败]"))
    print(f"最终统计: {len(entries)} 条, 失败: {fails}")


if __name__ == "__main__":
    main()
