"""
测试卡牌完整文本聚合与效果完整性

测试卡牌：「马纳历亚的亲爱挚友·安＆古蕾娅」

检查项：
1. 是否命中卡牌
2. 是否聚合完整文本
3. 聚合文本中是否包含：疾驰、结束阶段、安的巨大英灵、入场曲、舍弃2张学院类型、5点伤害、抽取2张卡

如果缺少任意一项，测试失败。
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'backend')

from retriever_local import get_retriever
from rag_pipeline_local import extract_card_names, _format_card_for_judge, build_prompt


def test_card_full_text():
    """测试「马纳历亚的亲爱挚友·安＆古蕾娅」的完整效果提取"""
    question = '现在告诉我当手牌只剩一张时，可以发动【马纳历亚的亲爱挚友·安＆古蕾娅】的入场曲么'

    print("=" * 60)
    print("测试：卡牌完整文本聚合")
    print("=" * 60)

    # 1. 提取卡名
    r = get_retriever()
    card_names = extract_card_names(question, r)
    print(f"\n[步骤1] 提取到的卡名: {card_names}")
    assert len(card_names) > 0, "FAIL: 未提取到卡名"
    print("  PASS: 卡名提取成功")

    # 2. 精确匹配
    matches = r.search_cards_by_name(card_names)
    print(f"\n[步骤2] 精确匹配结果数: {len(matches)}")
    assert len(matches) > 0, "FAIL: 未匹配到卡牌"
    print("  PASS: 卡牌匹配成功")

    # 3. 检查标记
    card = matches[0]
    print(f"\n[步骤3] 卡牌标记检查:")
    print(f"  name_cn: {card.get('name_cn')}")
    print(f"  is_exact_card_match: {card.get('is_exact_card_match')}")
    print(f"  is_full_card_text: {card.get('is_full_card_text')}")
    assert card.get('is_exact_card_match') == True, "FAIL: is_exact_card_match 不为 True"
    assert card.get('is_full_card_text') == True, "FAIL: is_full_card_text 不为 True"
    print("  PASS: 标记正确")

    # 4. 格式化输出
    print(f"\n[步骤4] 格式化卡牌文本:")
    formatted = _format_card_for_judge(card['name_cn'], card['text_preview'])
    print(formatted)

    # 5. 检查关键内容
    print(f"\n[步骤5] 关键内容检查:")
    checks = [
        ("疾驰", "疾驰能力"),
        ("结束阶段", "结束阶段能力"),
        ("安的巨大英灵", "安的巨大英灵召唤"),
        ("入场曲", "入场曲能力"),
        ("舍弃", "舍弃2张学院类型"),
        ("5点伤害", "5点伤害"),
        ("抽取2张卡", "抽取2张卡"),
    ]
    all_pass = True
    for keyword, desc in checks:
        found = keyword in formatted
        status = "PASS" if found else "FAIL"
        if not found:
            all_pass = False
        print(f"  [{status}] 包含「{keyword}」({desc}): {found}")

    assert all_pass, "FAIL: 缺少关键效果文本"
    print("  ALL PASS: 所有关键内容均存在")

    # 6. build_prompt 验证
    print(f"\n[步骤6] Prompt 组装验证:")
    sources = r.search(question, top_k=10)
    prompt = build_prompt(question, sources, matches)
    assert "入场曲" in prompt, "FAIL: Prompt 中缺少入场曲"
    assert "舍弃" in prompt, "FAIL: Prompt 中缺少舍弃"
    assert "2张" in prompt, "FAIL: Prompt 中缺少2张"
    print("  PASS: Prompt 中包含完整效果文本")

    # 7. sources 格式验证（仅检索层）
    print(f"\n[步骤7] Sources 格式验证（仅检索层）:")
    card_names = extract_card_names(question, r)
    exact_card_matches = r.search_cards_by_name(card_names)
    sources = r.search(question, top_k=20)

    # 合并
    if exact_card_matches:
        exact_names = {c.get("name_cn", "") for c in exact_card_matches}
        card_sources_from_vec = [c for c in sources if c.get("source_type") == "cards"]
        card_sources = exact_card_matches + [
            c for c in card_sources_from_vec
            if c.get("name_cn", "") not in exact_names
        ]
    else:
        card_sources = [c for c in sources if c.get("source_type") == "cards"]

    all_sources = exact_card_matches + sources
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
        if s.get("source_type") == "cards":
            source_entry["is_exact_card_match"] = s.get("is_exact_card_match", False)
            source_entry["is_full_card_text"] = s.get("is_full_card_text", False)
            if not s.get("is_full_card_text", True):
                source_entry["warning"] = "未能聚合完整卡牌文本"
        formatted_sources.append(source_entry)

    card_sources_formatted = [s for s in formatted_sources if s.get('source_type') == 'cards']
    exact_cards = [s for s in card_sources_formatted if s.get('is_exact_card_match')]

    print(f"  总 sources: {len(formatted_sources)}")
    print(f"  卡牌 sources: {len(card_sources_formatted)}")
    print(f"  精确匹配卡牌: {len(exact_cards)}")

    if exact_cards:
        ec = exact_cards[0]
        print(f"  is_exact_card_match: {ec.get('is_exact_card_match')}")
        print(f"  is_full_card_text: {ec.get('is_full_card_text')}")
        print(f"  text_preview 长度: {len(ec.get('text_preview', ''))}")
        has_fanfare = '入场曲' in ec.get('text_preview', '')
        print(f"  text_preview 包含入场曲: {has_fanfare}")
        assert ec.get('is_exact_card_match') == True
        assert ec.get('is_full_card_text') == True
        assert has_fanfare, "FAIL: text_preview 中缺少入场曲"
        print("  PASS: Sources 格式正确，包含完整效果")
    else:
        print("  FAIL: 没有精确匹配的卡牌 source")
        assert False

    print(f"\n{'=' * 60}")
    print("所有测试通过！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_card_full_text()
