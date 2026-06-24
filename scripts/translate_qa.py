#!/usr/bin/env python3
"""
影之诗进化对决 QA 数据集中日翻译脚本 v2
优化版：每次 API 调用批量翻译多个条目，大幅提升效率。

数据来源:
  1. canonical_terms.json - 官方术语映射
  2. sve_card_full_data.csv - 日文卡牌名称→中文卡牌名称映射
  3. DeepSeek API - 最终翻译

翻译优先级:
  优先级1: canonical_terms.json 术语映射
  优先级2: sve_card_full_data.csv 卡牌名称映射
  优先级3: DeepSeek 模型自主翻译
"""

import json
import csv
import time
import re
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = "sk-2e776b679daa4bf581324cf13c9393d6"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 路径配置
WORKSPACE = Path(r"D:\code\file\sveruler workbuddy")
INPUT_FILE = WORKSPACE / "rag_chunks" / "qa_chunks.jsonl"
OUTPUT_FILE = WORKSPACE / "rag_chunks" / "qa_chunks_cn.jsonl"
TERMS_FILE = WORKSPACE / "canonical_terms.json"
CARDS_FILE = WORKSPACE / "sve_card_full_data.csv"

# 每批翻译的条目数（单次 API 调用）
BATCH_SIZE = 15
# API 调用间延迟（秒）
API_DELAY = 1.5
# 最大重试次数
MAX_RETRIES = 3

# ============================================================
# 术语映射构建
# ============================================================

def build_term_mapping(terms_file: Path) -> Dict[str, str]:
    """从 canonical_terms.json 构建日文→中文术语映射"""
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms_data = json.load(f)

    jp_to_cn = {}
    for concept_id, term_info in terms_data.items():
        canonical_cn = term_info.get("canonical_cn", "")
        jp = term_info.get("jp", "")
        if jp and canonical_cn:
            jp_to_cn[jp] = canonical_cn
        for alias_jp in term_info.get("aliases_jp", []):
            if alias_jp and canonical_cn:
                jp_to_cn[alias_jp] = canonical_cn

    # 按长度降序排列，确保长术语优先匹配
    return dict(sorted(jp_to_cn.items(), key=lambda x: len(x[0]), reverse=True))


def build_card_mapping(cards_file: Path) -> Dict[str, str]:
    """从 CSV 构建日文卡牌名称→中文卡牌名称映射"""
    jp_to_cn = {}
    with open(cards_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_jp = row.get('name_jp', '').strip()
            name_cn = row.get('name_cn', '').strip()
            if name_jp and name_cn:
                jp_to_cn[name_jp] = name_cn
    return dict(sorted(jp_to_cn.items(), key=lambda x: len(x[0]), reverse=True))


def build_term_table(term_map: Dict[str, str], max_terms: int = 120) -> str:
    """构建术语表字符串，用于注入到 prompt 中"""
    core_keys = [
        "ファンファーレ", "ラストワード", "進化", "超進化", "エボルヴ",
        "クイック", "守護", "疾走", "突進", "起動", "ドレイン",
        "フォロワー", "スペル", "アミュレット", "トークン", "イクイップメント",
        "プレイヤー", "リーダー", "ターン", "メインフェイズ", "エンドフェイズ",
        "スタートフェイズ", "先攻", "後攻", "バトル", "ゲーム",
        "デッキ", "メインデッキ", "エボルヴデッキ", "EXエリア",
        "手札", "墓場", "場", "消滅", "破壊", "捨てる",
        "攻撃力", "体力", "コスト", "PP", "EP", "SEP",
        "ダメージ", "選択", "チョイス", "アクト", "スタンド",
        "アドバンス", "カード名",
        "公開領域", "非公開領域", "リーダーエリア",
        "ドライブ領域", "解決領域", "トリガー領域", "出走領域",
        "イクイップメント領域", "進化領域", "消滅領域",
        "進化時", "超進化時", "攻撃時", "登場時",
        "付帯条項", "フレーバー", "イラスト", "レアリティ",
        "クラス", "ニュートラル", "エルフ", "ロイヤル", "ウィッチ",
        "ドラゴン", "ナイトメア", "ビショップ",
        "選ぶ", "割り振る", "関連付ける", "消滅する", "破壊する",
        "表向き", "裏向き", "敗北", "勝利", "引き分け",
        "対戦相手", "非ターンプレイヤー", "ターンプレイヤー",
        "オーナー", "マスター", "タイプ", "特殊な種類",
        "デッキ置き場", "エボルヴデッキ置き場",
        "カードを引く", "配置状態", "アクト状態", "スタンド状態",
        "進化した", "勝敗", "経過ターン数",
        "PP最大値", "投了", "EPを得る", "SEPを得る",
        "両面カード", "クレスト", "トークン情報",
        "コラボ名", "タイトル", "イラスト",
        "テキスト", "カードテキスト", "特殊な種類",
        # 动词形式
        "アクトする", "進化する", "消滅する", "破壊する",
        "捨てる", "選ぶ", "割り振る", "関連付ける",
    ]

    table = []
    seen = set()

    # 先添加核心术语
    for key in core_keys:
        if key in term_map:
            cn = term_map[key]
            if cn not in seen:
                table.append(f"{key} → {cn}")
                seen.add(cn)

    # 补充其他术语
    for jp, cn in term_map.items():
        if len(table) >= max_terms:
            break
        if cn not in seen and len(jp) >= 2:
            table.append(f"{jp} → {cn}")
            seen.add(cn)

    return "\n".join(table)


def build_system_prompt(term_map: Dict[str, str]) -> str:
    """构建系统翻译提示"""
    term_table = build_term_table(term_map)

    return f"""你是影之诗进化对决（Shadowverse EVOLVE）卡牌游戏的官方中文翻译专家。
请将多个日文 QA 条目批量翻译为中文，严格遵循规则。

【术语表 - 必须严格使用】
{term_table}

【翻译规则】
1. 游戏关键字保留格式：《ファンファーレ》→《入场曲》、【守護】→【守护】、【疾走】→【疾驰】
2. 卡牌名称使用书名号「」包裹（如『ダークエンジェル・オリヴィエ』→「暗黑天使·奥莉薇」）
3. 保留所有数字、费用（コスト）、攻击力、防御力（体力）、回合数不变
4. 保留规则判定逻辑，不改变原意
5. 不省略任何句子，完整翻译
6. 「はい」→「是」，「いいえ」→「否」
7. 语法自然中文表达

【输出格式 - 严格遵循】
对于每个条目，输出格式：
---ENTRY id---
Q_cn: <中文问题>
A_cn: <中文答案>

每个条目之间用 ---ENTRY id--- 分隔。id 必须与输入的条目 id 一致。"""


# ============================================================
# DeepSeek 批量翻译
# ============================================================

def translate_batch(
    client: OpenAI,
    entries: List[dict],
    term_map: Dict[str, str],
    card_map: Dict[str, str],
) -> List[dict]:
    """
    一次 API 调用批量翻译多个 QA 条目。
    将所有条目打包为单个 prompt，解析返回的结果。
    """
    if not entries:
        return []

    # 生成序号（用于精确匹配）
    entry_map = {}
    prompt_parts = []
    for i, entry in enumerate(entries):
        entry_map[i] = entry
        # 预替换术语和卡名（作为上下文辅助）
        q_pre = apply_all_substitutions(entry["question"], term_map, card_map)
        a_pre = apply_all_substitutions(entry["answer"], term_map, card_map)

        prompt_parts.append(
            f"[条目{i}] id={entry['id']}\n"
            f"Q: {entry['question']}\n"
            f"A: {entry['answer']}"
        )

    prompt_text = "\n\n".join(prompt_parts)

    user_prompt = f"""请将以下 {len(entries)} 个影之诗进化对决的日文 QA 条目翻译为中文。

【上下文辅助（预替换的术语和卡名）】
以下为部分术语的预替换版本，仅供参考，请以术语表为准进行翻译：

"""
    for i, entry in enumerate(entries):
        q_pre = apply_all_substitutions(entry["question"], term_map, card_map)
        a_pre = apply_all_substitutions(entry["answer"], term_map, card_map)
        user_prompt += f"[条目{i}] 预替换版:\nQ: {q_pre}\nA: {a_pre}\n\n"

    user_prompt += f"""【原始日文 - 以此为翻译源】
{prompt_text}

请按格式输出每个条目的翻译：
---ENTRY <条目id>---
Q_cn: <中文问题>
A_cn: <中文答案>

注意：
1. 必须使用 ---ENTRY <id>--- 作为分隔符
2. id 必须与输入完全一致
3. 卡牌名如能识别中文名则翻译，否则保留日文原名"""

    system_prompt = build_system_prompt(term_map)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=8192,
            )

            result_text = response.choices[0].message.content.strip()
            parsed = parse_batch_result(result_text, entries)

            # 检查是否有缺失的条目
            parsed_ids = {e["id"] for e in parsed}
            input_ids = {e["id"] for e in entries}
            missing = input_ids - parsed_ids

            if missing:
                print(f"  [警告] 解析遗漏 {len(missing)} 个条目: {missing}")
                # 补充遗漏条目
                for entry in entries:
                    if entry["id"] in missing:
                        entry["Q_cn"] = f"[解析遗漏] {entry['question']}"
                        entry["A_cn"] = f"[解析遗漏] {entry['answer']}"
                        parsed.append(entry)

            time.sleep(API_DELAY)
            return parsed

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 10
                print(f"  [重试] 批次翻译失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}. 等待 {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [失败] 批次翻译完全失败: {e}")
                # 降级：逐条翻译
                return fallback_translate_batch(client, entries, term_map, card_map)

    return entries


def apply_all_substitutions(text: str, term_map: Dict[str, str], card_map: Dict[str, str]) -> str:
    """对文本应用术语和卡名的预替换"""
    result = text
    # 先替换卡名（在『』中的）
    def replace_card(match):
        jp_name = match.group(1)
        if jp_name in card_map:
            return f'「{card_map[jp_name]}」'
        return match.group(0)

    result = re.sub(r'『([^』]+)』', replace_card, result)

    # 再替换术语
    for jp_term, cn_term in term_map.items():
        if len(jp_term) >= 2:  # 只替换>=2字符的术语，避免短词误伤
            result = result.replace(jp_term, cn_term)

    return result


def parse_batch_result(result_text: str, entries: List[dict]) -> List[dict]:
    """解析批量翻译结果"""
    results = []
    entry_map = {e["id"]: e for e in entries}

    # 按 ---ENTRY <id>--- 分割
    pattern = r'---ENTRY\s+([^\n-]+?)\s*(?:---)?\s*\n(.*?)(?=---ENTRY|\Z)'
    matches = list(re.finditer(pattern, result_text, re.DOTALL))

    if not matches:
        # 备选：按 ---ENTRY 分割
        parts = re.split(r'---ENTRY\s*', result_text)
        for part in parts[1:]:  # 跳过开头空白
            lines = part.strip().split('\n', 1)
            if len(lines) >= 1:
                entry_id = lines[0].strip().rstrip('-').strip()
                content = lines[1] if len(lines) > 1 else ""
                if entry_id in entry_map:
                    entry = dict(entry_map[entry_id])
                    q_cn, a_cn = extract_q_a(content)
                    entry["Q_cn"] = q_cn
                    entry["A_cn"] = a_cn
                    results.append(entry)
    else:
        for match in matches:
            entry_id = match.group(1).strip().rstrip('-').strip()
            content = match.group(2)

            if entry_id in entry_map:
                entry = dict(entry_map[entry_id])
                q_cn, a_cn = extract_q_a(content)
                entry["Q_cn"] = q_cn
                entry["A_cn"] = a_cn
                results.append(entry)

    # 对未匹配到的条目，尝试全文搜索
    matched_ids = {r["id"] for r in results}
    for entry in entries:
        if entry["id"] not in matched_ids:
            # 在结果中搜索条目 id
            eid = entry["id"]
            if eid in result_text:
                idx = result_text.index(eid)
                surrounding = result_text[idx:idx+1000]
                q_cn, a_cn = extract_q_a(surrounding)
                entry_cp = dict(entry)
                entry_cp["Q_cn"] = q_cn
                entry_cp["A_cn"] = a_cn
                results.append(entry_cp)

    return results


def extract_q_a(content: str) -> Tuple[str, str]:
    """从内容中提取 Q_cn 和 A_cn，支持多种格式"""
    q_cn = ""
    a_cn = ""

    # 模式1: Q_cn: xxx / A_cn: xxx
    q_match = re.search(r'Q_cn\s*[:：]\s*(.+?)(?=\nA_cn\s*[:：]|\nA:|\n---|\Z)', content, re.DOTALL)
    if q_match:
        q_cn = q_match.group(1).strip()

    a_match = re.search(r'A_cn\s*[:：]\s*(.+?)(?=\n(?:Q_cn|---|\Z))', content, re.DOTALL)
    if a_match:
        a_cn = a_match.group(1).strip()

    # 模式2: 问题：xxx / 答案：xxx
    if not q_cn:
        q_match = re.search(r'(?:问题|中文问题)\s*[:：]\s*(.+?)(?=\n(?:答案|中文答案|A_cn)\s*[:：]|\n---|\Z)', content, re.DOTALL)
        if q_match:
            q_cn = q_match.group(1).strip()
    if not a_cn:
        a_match = re.search(r'(?:答案|中文答案)\s*[:：]\s*(.+?)(?=\n(?:问题|中文问题|Q_cn|---|\Z))', content, re.DOTALL)
        if a_match:
            a_cn = a_match.group(1).strip()

    # 模式3: Q: xxx / A: xxx
    if not q_cn:
        q_match = re.search(r'(?<!\w)Q\s*[:：]\s*(.+?)(?=\nA\s*[:：]|\n---|\Z)', content, re.DOTALL)
        if q_match:
            q_cn = q_match.group(1).strip()
    if not a_cn:
        a_match = re.search(r'(?<!\w)A\s*[:：]\s*(.+?)(?=\nQ\s*[:：]|\n---|\Z)', content, re.DOTALL)
        if a_match:
            a_cn = a_match.group(1).strip()

    # 模式4: 逐行解析 - Q_cn 和 A_cn 各自独占一行
    if not q_cn or not a_cn:
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not q_cn and re.match(r'Q_cn\s*[:：]', line):
                # 收集后续行直到遇到 A_cn 或空行
                parts = [re.sub(r'^Q_cn\s*[:：]\s*', '', line)]
                for j in range(i+1, len(lines)):
                    nl = lines[j].strip()
                    if re.match(r'A_cn\s*[:：]', nl):
                        break
                    parts.append(nl)
                q_cn = ' '.join(parts).strip()
            if not a_cn and re.match(r'A_cn\s*[:：]', line):
                parts = [re.sub(r'^A_cn\s*[:：]\s*', '', line)]
                for j in range(i+1, len(lines)):
                    nl = lines[j].strip()
                    if re.match(r'Q_cn\s*[:：]', nl):
                        break
                    parts.append(nl)
                a_cn = ' '.join(parts).strip()

    return q_cn, a_cn


def fallback_translate_batch(
    client: OpenAI,
    entries: List[dict],
    term_map: Dict[str, str],
    card_map: Dict[str, str],
) -> List[dict]:
    """降级方案：逐条翻译"""
    print("  [降级] 切换到逐条翻译模式...")
    results = []
    system_prompt = build_system_prompt(term_map)

    for entry in entries:
        for attempt in range(MAX_RETRIES):
            try:
                q_pre = apply_all_substitutions(entry["question"], term_map, card_map)
                a_pre = apply_all_substitutions(entry["answer"], term_map, card_map)

                prompt = f"""翻译这个影之诗 QA 条目：

日文问题：{entry['question']}

日文答案：{entry['answer']}

预替换参考：
问题：{q_pre}
答案：{a_pre}

输出格式：
Q_cn: <中文问题>
A_cn: <中文答案>"""

                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

                result = response.choices[0].message.content.strip()
                q_cn, a_cn = extract_q_a(result)

                entry_cp = dict(entry)
                entry_cp["Q_cn"] = q_cn if q_cn else f"[翻译失败] {entry['question']}"
                entry_cp["A_cn"] = a_cn if a_cn else f"[翻译失败] {entry['answer']}"
                results.append(entry_cp)
                time.sleep(API_DELAY)
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep((attempt + 1) * 5)
                else:
                    entry_cp = dict(entry)
                    entry_cp["Q_cn"] = f"[翻译失败] {entry['question']}"
                    entry_cp["A_cn"] = f"[翻译失败] {entry['answer']}"
                    results.append(entry_cp)

    return results


# ============================================================
# Embedding 更新
# ============================================================

def update_embedding(entry: dict) -> str:
    """更新 embedding_text 为中日双语格式"""
    orig = entry.get("embedding_text", "")

    cn_part = f"""
Q_CN:
{entry.get('Q_cn', '')}

A_CN:
{entry.get('A_cn', '')}"""

    return orig + cn_part


# ============================================================
# 文件操作
# ============================================================

def load_entries(filepath: Path) -> List[dict]:
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_entries(entries: List[dict], filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_checkpoint(filepath: Path) -> List[dict]:
    """加载已有的翻译进度"""
    if filepath.exists():
        return load_entries(filepath)
    return []


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("影之诗进化对决 QA 中日翻译系统 v2")
    print("=" * 60)

    # Step 1: 加载映射
    print("\n[1/4] 加载术语和卡牌映射...")
    term_map = build_term_mapping(TERMS_FILE)
    card_map = build_card_mapping(CARDS_FILE)
    print(f"  术语: {len(term_map)} 条, 卡名: {len(card_map)} 条")

    # Step 2: 加载 QA 数据
    print("\n[2/4] 加载 QA 数据...")
    entries = load_entries(INPUT_FILE)
    print(f"  QA 条目: {len(entries)}")

    # 检查断点续传
    checkpoint = load_checkpoint(OUTPUT_FILE)
    if checkpoint:
        completed_ids = {e["id"] for e in checkpoint if e.get("Q_cn") and not e.get("Q_cn", "").startswith("[翻译失败]")}
        if completed_ids:
            print(f"  检测到已完成 {len(completed_ids)} 条，跳过...")
            entries = [e for e in entries if e["id"] not in completed_ids]
            print(f"  剩余待翻译: {len(entries)}")
    else:
        checkpoint = []

    # Step 3: 初始化客户端
    print("\n[3/4] 初始化 DeepSeek API...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Step 4: 批量翻译
    total_batches = (len(entries) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[4/4] 开始批量翻译 ({len(entries)} 条, {total_batches} 批, 每批 {BATCH_SIZE} 条)...")

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(entries))
        batch = entries[start:end]

        print(f"\n  批次 {batch_idx + 1}/{total_batches} (条目 {start+1}-{end})...", end=" ", flush=True)

        translated = translate_batch(client, batch, term_map, card_map)

        # 更新 embedding
        for entry in translated:
            entry["embedding_text"] = update_embedding(entry)

        # 合并入 checkpoint
        checkpoint.extend(translated)
        save_entries(checkpoint, OUTPUT_FILE)

        success_count = sum(1 for e in translated if e.get("Q_cn") and not e["Q_cn"].startswith("[翻译失败]") and not e["Q_cn"].startswith("[解析遗漏]"))
        print(f"完成 ({success_count}/{len(batch)} 成功)")

    # 完成统计
    print("\n" + "=" * 60)
    final = load_entries(OUTPUT_FILE)
    total = len(final)
    ok = sum(1 for e in final if e.get("Q_cn") and not e["Q_cn"].startswith(("[翻译失败]", "[解析遗漏]", "[降级")))
    fail = total - ok
    print(f"输出: {OUTPUT_FILE}")
    print(f"总计: {total}, 成功: {ok}, 需修复: {fail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
