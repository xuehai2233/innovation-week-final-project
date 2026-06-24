"""
影之诗进化对决 AI裁判 —— RAG 管线
二阶段检索: 卡名精确匹配 → 向量检索 → Prompt 组装 → LLM 调用 → 返回
"""
import re
from retriever_local import get_retriever
from local_llm import LocalLLM

# 系统提示词
SYSTEM_PROMPT = """你是《影之诗进化对决》(Shadowverse EVOLVE) 卡牌游戏的官方裁判助手。

你的职责是回答玩家关于游戏规则、卡牌效果、战斗机制等问题的裁定。

回答规则:
1. 优先依据提供的"官方QA"资料，这是最权威的裁定来源。
2. 如果没有匹配的官方QA，则依据"综合规则"和"卡牌文本"进行推理。
3. 如果现有资料不足以做出确定裁定，必须说明"当前资料中没有找到针对此问题的直接官方裁定"，并给出基于现有规则的最合理推测。
4. 绝对不允许编造不存在的官方QA或规则条文。
5. 【重要】在分析涉及卡牌的问题时，必须优先使用【相关卡牌】中提供的卡牌数据（效果文本、费用、攻击力/生命值、关键词等），不要依赖猜测。

回答格式:
裁定：（一句话给出明确裁定结论）

依据：（引用具体的规则编号或卡牌效果原文）

解释：（详细解释裁定逻辑）

结论：（总结性结论）"""

# 卡名匹配模式：支持「」『』【】《》"" '' 以及直接中文书名号内的文本
CARD_NAME_PATTERNS = [
    re.compile(r'[「『【《]([^」』】》\n]{1,30})[」』】》]'),
    re.compile(r'["\']([^"\'\n]{1,30})["\']'),
]


def extract_card_names(question, retriever=None):
    """
    从玩家问题中提取卡牌名称

    策略（多级匹配）:
    1. 正则匹配「」『』【】《》"" '' 中的文本
    2. 常见卡名模式（如"XX·XX"格式）
    3. （如果提供 retriever）用已知卡名在问题中做子串扫描——处理无括号嵌入的情况
    4. 去重后返回
    """
    names = []

    for pattern in CARD_NAME_PATTERNS:
        matches = pattern.findall(question)
        for m in matches:
            m = m.strip()
            # 过滤太短或明显不是卡名的（如纯数字、单字等）
            if len(m) >= 2 and not m.isdigit():
                names.append(m)

    # 子串扫描 fallback：用已知卡名在问题中扫描
    # 处理"马纳历亚的亲爱挚友·安＆古蕾娅"这种无括号嵌入的卡名
    if retriever is not None:
        known_names = list(retriever._card_bundles.keys())
        # 按卡名长度降序排列，优先匹配长卡名（避免"安"匹配到"安＆古蕾娅"的一部分）
        known_names.sort(key=len, reverse=True)
        for known_name in known_names:
            if len(known_name) < 3:
                continue
            if known_name in question and known_name not in names:
                names.append(known_name)

    # 去重，保持顺序
    seen = set()
    unique_names = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)

    return unique_names


def expand_query(query):
    """简单 query expansion：添加关键词变体"""
    terms = {
        "进化": "进化 进化点 进化后",
        "守护": "守护 Ward 守护随从",
        "攻击": "攻击 攻击力 ATK 战斗伤害",
        "手牌": "手牌 手札 手牌数",
        "主战者": "主战者 玩家 Leader",
        "抽牌": "抽牌 抽卡 Draw ドロー",
        "破坏": "破坏 消灭 破坏效果",
        "回复": "回复 恢复 回血",
        "费用": "费用 PP 法力 cost 消耗",
        "墓地": "墓地 坟场 Cemetery",
        "消失": "消失 banish 除外",
        "入场曲": "入场曲 登场时 战吼 fanfare",
        "谢幕曲": "谢幕曲 离场时 亡语 lastwords",
    }

    expanded = [query]
    for key, val in terms.items():
        if key in query:
            expanded.append(val)
    return " ".join(expanded)


def _extract_multiline_field(embedding_text, field_name):
    """
    从 embedding_text 中提取字段值，支持多行值（如中文效果可能跨多行）

    策略：
    1. 从 field_name 后开始提取
    2. 一直读取到下一个已知字段名或文本末尾
    3. 返回完整的多行值
    """
    # 已知的后续字段名（按顺序）
    known_fields = [
        "中文名", "日文名", "职业", "卡牌种类", "类型", "费用",
        "攻击力", "生命值", "稀有度", "出处",
        "中文效果", "日文效果", "关联卡", "关键词",
        "卡号",  # 可能在前面
    ]

    # 找到 field_name 的起始位置
    pattern = field_name + r'[：:]'
    m = re.search(pattern, embedding_text)
    if not m:
        return ""

    start = m.end()
    remaining = embedding_text[start:]

    # 找到下一个字段名的位置
    next_field_pos = len(remaining)
    for f in known_fields:
        if f == field_name:
            continue
        # 匹配 "字段名：" 或 "字段名:" 模式
        fpattern = f + r'[：:]'
        fm = re.search(fpattern, remaining)
        if fm and fm.start() < next_field_pos:
            next_field_pos = fm.start()

    value = remaining[:next_field_pos].strip()
    return value


def _format_card_for_judge(name, embedding_text):
    """
    从卡牌 embedding_text 中提取裁判需要的关键信息

    返回完整格式，包含所有效果文本，不再截断。

    embedding_text 格式示例:
      卡号：BP01-089
      中文名：灼热风暴
      职业：Dragon
      卡牌种类：Spell
      费用：5
      中文效果：给予全体从者各5点伤害。
      关键词：Dragon Spell 伤害 从者

    返回格式:
      灼热风暴 / マナリアフレンズ・アン＆グレア
      类型: Follower · Rune · 魔法使い・学院・プリンセス
      费用: 5 | 攻击力: 4 | 生命值: 4
      效果:
      1. 只要自己的墓场中的学院类型·卡片有10张及以上存在...
      2. 当自己的结束阶段到来时...
      3. 《入场曲》将手牌中的2张学院类型·卡片舍弃...
    """
    if not embedding_text:
        return name

    # 提取各字段（效果类字段用多行提取）
    def extract_single(field_name):
        """提取单行字段"""
        pattern = field_name + r'[：:]([^\n]+)'
        m = re.search(pattern, embedding_text)
        return m.group(1).strip() if m else ""

    card_no = extract_single("卡号")
    name_jp = extract_single("日文名")
    card_type = extract_single("卡牌种类")
    craft = extract_single("职业")
    type_info_raw = extract_single("类型")
    cost = extract_single("费用")
    attack = extract_single("攻击力")
    life = extract_single("生命值")
    rare = extract_single("稀有度")
    set_name = extract_single("出处")

    # 效果字段使用多行提取
    effect_cn = _extract_multiline_field(embedding_text, "中文效果")
    effect_jp = _extract_multiline_field(embedding_text, "日文效果")
    keywords = _extract_multiline_field(embedding_text, "关键词")

    lines = []

    # 卡名 + 日文名
    name_line = name
    if name_jp:
        name_line += f" / {name_jp}"
    lines.append(name_line)

    # 卡号
    if card_no:
        lines.append(f"卡号: {card_no}")

    # 类型信息
    type_parts = []
    if card_type:
        type_parts.append(card_type)
    if craft:
        type_parts.append(craft)
    if type_info_raw:
        type_parts.append(type_info_raw)
    if type_parts:
        lines.append(f"类型: {' · '.join(type_parts)}")

    # 费用 + 攻/血
    stat_parts = []
    if cost and cost != "-1":
        stat_parts.append(f"费用: {cost}")
    if card_type in ("Follower", "FollowerEvo", "随从") or (attack and attack != "-1"):
        atk = attack if attack and attack != "-1" else "?"
        hp = life if life and life != "-1" else "?"
        stat_parts.append(f"攻击力: {atk}")
        stat_parts.append(f"生命值: {hp}")
    if stat_parts:
        lines.append(" | ".join(stat_parts))

    # 稀有度 + 出处
    meta_parts = []
    if rare:
        meta_parts.append(f"稀有度: {rare}")
    if set_name:
        meta_parts.append(f"出处: {set_name}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))

    # 效果文本（完整，按换行拆分编号）
    if effect_cn:
        # 将效果文本按换行拆分，去掉空行，编号
        effect_lines = [l.strip() for l in effect_cn.split('\n') if l.strip()]
        if len(effect_lines) == 1:
            lines.append(f"效果: {effect_lines[0]}")
        else:
            lines.append("效果:")
            for idx, el in enumerate(effect_lines, 1):
                lines.append(f"  {idx}. {el}")

    # 关键词
    if keywords:
        lines.append(f"关键词: {keywords}")

    return "\n".join(lines)


def _detect_card_question_focus(question):
    """
    检测问题关注的卡牌状态，决定进化前/后的展示顺序

    返回: "evo_first" | "base_first" | "balanced"
    """
    evo_keywords = ["进化后", "进化时", "超进化", "Evo", "灵气", "进化能力"]
    base_keywords = ["入场曲", "UB", "召唤时", "登场时", "fanfare", "Fanfare"]

    focus_evo = any(kw in question for kw in evo_keywords)
    focus_base = any(kw in question for kw in base_keywords)

    if focus_evo and not focus_base:
        return "evo_first"
    elif focus_base and not focus_evo:
        return "base_first"
    return "balanced"


def build_prompt(question, sources, card_bundles=None):
    """
    组装裁判 prompt

    参数:
      question: 玩家问题
      sources: 向量检索结果
      card_bundles: CardBundle 列表（卡名精确匹配结果，含进化前/后完整信息）
    """
    # 分类 sources
    qa_sources = [s for s in sources if s.get("source_type") == "qa"]
    rule_sources = [s for s in sources if s.get("source_type") == "rules"]
    section_sources = [s for s in sources if s.get("source_type") == "sections"]
    vec_card_sources = [s for s in sources if s.get("source_type") == "cards"]

    # 去重：已在 bundle 中的卡名，从向量检索卡牌中移除
    if card_bundles:
        bundle_names = {b["name_cn"] for b in card_bundles}
        vec_card_sources = [c for c in vec_card_sources if c.get("name_cn", "") not in bundle_names]

    parts = []

    # 玩家问题
    parts.append(f"玩家提问：{question}")
    parts.append("")

    # 卡牌文本（放在最前面，让 LLM 优先参考卡牌效果）
    if card_bundles or vec_card_sources:
        parts.append("【相关卡牌】")
        parts.append(
            "注意：以下卡牌文本为完整效果文本，请阅读全部效果后再裁定。"
            "部分随从卡同时展示进化前(Follower)和进化后(FollowerEvo)效果——"
            "两者效果完全不同，都需仔细阅读。"
            "不要只根据部分效果判断该卡没有其他能力（如入场曲、UB等）。"
        )
        focus = _detect_card_question_focus(question) if card_bundles else "balanced"

        # 精确匹配的 bundles（进化前/后分别展示）
        if card_bundles:
            card_idx = 0
            for bundle in card_bundles[:15]:
                card_idx += 1
                name = bundle["name_cn"]
                follower = bundle.get("follower")
                follower_evo = bundle.get("follower_evo")
                follower_adv = bundle.get("follower_adv")
                spell = bundle.get("spell")
                spell_evo = bundle.get("spell_evo")
                amulet = bundle.get("amulet")
                amulet_evo = bundle.get("amulet_evo")

                # 非随从卡（Spell/Amulet/Loader 等）：直接展示
                if not follower and not follower_evo and not follower_adv:
                    if spell:
                        parts.append(f"--- 卡牌{card_idx}：{name} ---")
                        parts.append(_format_card_for_judge(name, spell["text_preview"]))
                    if spell_evo:
                        parts.append(f"（进化牌组专用法术）")
                        parts.append(_format_card_for_judge(name, spell_evo["text_preview"]))
                    if amulet:
                        parts.append(f"--- 卡牌{card_idx}：{name} ---")
                        parts.append(_format_card_for_judge(name, amulet["text_preview"]))
                    if amulet_evo:
                        parts.append(f"（进化后护符）")
                        parts.append(_format_card_for_judge(name, amulet_evo["text_preview"]))
                    continue

                # 随从卡：分别展示进化前/后
                parts.append(f"--- 卡牌{card_idx}：{name} ---")

                if follower and follower_evo:
                    # 根据问题焦点决定展示顺序
                    if focus == "evo_first":
                        parts.append("=== 进化后 (FollowerEvo) ===")
                        parts.append(_format_card_for_judge(name, follower_evo["text_preview"]))
                        parts.append("=== 进化前 (Follower) ===")
                        parts.append(_format_card_for_judge(name, follower["text_preview"]))
                    else:
                        parts.append("=== 进化前 (Follower) ===")
                        parts.append(_format_card_for_judge(name, follower["text_preview"]))
                        parts.append("=== 进化后 (FollowerEvo) ===")
                        parts.append(_format_card_for_judge(name, follower_evo["text_preview"]))
                elif follower_adv and follower_evo:
                    if focus == "evo_first":
                        parts.append("=== 进化后 (FollowerEvo) ===")
                        parts.append(_format_card_for_judge(name, follower_evo["text_preview"]))
                        parts.append("=== 高阶随从 (FollowerAdv) ===")
                        parts.append(_format_card_for_judge(name, follower_adv["text_preview"]))
                    else:
                        parts.append("=== 高阶随从 (FollowerAdv) ===")
                        parts.append(_format_card_for_judge(name, follower_adv["text_preview"]))
                        parts.append("=== 进化后 (FollowerEvo) ===")
                        parts.append(_format_card_for_judge(name, follower_evo["text_preview"]))
                elif follower:
                    parts.append(_format_card_for_judge(name, follower["text_preview"]))
                elif follower_evo:
                    parts.append("（仅进化后形态，无进化前数据）")
                    parts.append(_format_card_for_judge(name, follower_evo["text_preview"]))
                elif follower_adv:
                    parts.append(_format_card_for_judge(name, follower_adv["text_preview"]))

            parts.append("")

        # 向量检索补充卡牌（无进化拆分，作为补充参考）
        if vec_card_sources:
            for i, s in enumerate(vec_card_sources[:5], 1):
                name = s.get("name_cn", s.get("card_no", ""))
                text = s.get("text_preview", "")
                parts.append(f"（向量检索）卡牌：{name}")
                parts.append(_format_card_for_judge(name, text))
            parts.append("")

    # 官方QA（最高优先级）
    if qa_sources:
        parts.append("【官方QA记录】")
        for i, s in enumerate(qa_sources[:5], 1):
            q = s.get("Q_cn", s.get("text_preview", ""))
            a = s.get("A_cn", "")
            parts.append(f"QA{i}：")
            parts.append(f"  问：{q}")
            parts.append(f"  答：{a}")
        parts.append("")

    # 综合规则
    if rule_sources:
        parts.append("【综合规则】")
        for i, s in enumerate(rule_sources[:5], 1):
            title = s.get("title", s.get("rule_number", ""))
            text = s.get("text_preview", "")
            parts.append(f"规则{i} ({title})：{text}")
        parts.append("")

    # 章节
    if section_sources:
        parts.append("【规则章节】")
        for i, s in enumerate(section_sources[:2], 1):
            title = s.get("title", s.get("text_preview", ""))
            parts.append(f"章节{i}：{title}")
        parts.append("")

    parts.append("请根据以上资料对玩家提问做出裁判裁定。")
    parts.append(
        "特别注意：如果【相关卡牌】中提供了某张卡牌的进化前/后效果文本，"
        "请直接引用对应部分，阅读全部效果后再裁定，不要猜测或编造。"
    )

    return "\n".join(parts)


class RAGPipeline:
    def __init__(self, model="deepseek-chat", llm=None):
        self.retriever = get_retriever()
        self.llm = llm if llm is not None else LocalLLM(model=model)

    def judge(self, question, return_sources=False):
        """
        执行二阶段 RAG 裁判流程

        阶段1: 提取卡名 → 精确匹配 → 获取 CardBundle（含进化前/后完整信息）
        阶段2: Query expansion → 向量检索 QA + 规则
        阶段3: 合并 → 组装 prompt → LLM 调用

        返回:
          {"answer": "裁定回答", "sources": [...]}
        """
        # ===== 阶段1：卡名提取 + 精确匹配（返回 CardBundle） =====
        card_names = extract_card_names(question, self.retriever)
        card_bundles = []
        if card_names:
            card_bundles = self.retriever.search_cards_by_name(card_names)

        # ===== 阶段2：Query expansion + 向量检索 =====
        expanded_q = expand_query(question)
        sources = self.retriever.search(expanded_q, top_k=20)

        # ===== 阶段3：合并 + Prompt 组装 + LLM =====
        prompt = build_prompt(question, sources, card_bundles)
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        # ===== 格式化返回 =====
        # 将 bundles 展开为 flat sources 用于 API 响应
        flat_card_sources = []
        for bundle in card_bundles:
            for key in ("follower", "follower_evo", "follower_adv",
                        "spell", "spell_evo", "amulet", "amulet_evo"):
                meta = bundle.get(key)
                if meta:
                    meta = dict(meta)
                    meta["is_exact_card_match"] = bundle.get("is_exact_card_match", False)
                    meta["is_full_card_text"] = True
                    flat_card_sources.append(meta)

        all_sources = flat_card_sources + sources

        formatted_sources = []
        seen = set()
        for s in all_sources:
            sid = s.get("card_no", s.get("id", ""))
            if sid in seen:
                continue
            seen.add(sid)

            source_entry = {
                "id": s.get("id", s.get("card_no", "")),
                "source_type": s.get("source_type", ""),
                "title": s.get("title", s.get("Q_cn", s.get("name_cn", ""))),
                "score": round(s.get("score", 0), 4),
                "text_preview": s.get("text_preview", "")[:500],
            }

            # 卡牌类型的 source 带上完整性标记
            if s.get("source_type") == "cards":
                source_entry["is_exact_card_match"] = s.get("is_exact_card_match", False)
                source_entry["is_full_card_text"] = s.get("is_full_card_text", False)
                if not s.get("is_full_card_text", True):
                    source_entry["warning"] = "未能聚合完整卡牌文本"

            formatted_sources.append(source_entry)

        result = {
            "answer": answer.strip(),
            "sources": formatted_sources,
            "matched_cards": card_names,  # 返回提取到的卡名，便于调试
        }

        return result


# 全局单例
_pipeline = None


def get_pipeline(model="deepseek-chat", llm=None):
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(model=model, llm=llm)
    return _pipeline
