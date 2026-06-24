"""
测试 CardBundle 重构：验证进化前/后卡牌同时返回、完整效果保留
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'backend')

from retriever_local import get_retriever
from rag_pipeline_local import extract_card_names, _format_card_for_judge, _detect_card_question_focus, build_prompt

# 测试卡牌列表（含不同 card_type）
TEST_CARDS = [
    "可可萝",               # Follower + FollowerEvo
    "马纳历亚的亲爱挚友·安＆古蕾娅",  # 仅 Follower（no evo in data）
    "灼热风暴",             # Spell，无进化
    "灭剑焰龙·进攻模式",    # 仅 FollowerEvo（Token卡）
    "佩可莉姆",             # Follower + FollowerEvo
]

# 测试问题
TEST_QUESTIONS = {
    "evo_focus": "【可可萝】进化后有什么能力？",
    "base_focus": "【可可萝】的入场曲是什么？",
    "balanced": "【可可萝】这张卡什么效果？",
    "no_bracket": "可以发动马纳历亚的亲爱挚友·安＆古蕾娅的入场曲么",
}


def test_card_bundles():
    """测试 CardBundle 结构完整性"""
    r = get_retriever()
    errors = []

    # ===== 测试1: 验证 bundle 结构 =====
    print("=" * 60)
    print("[测试1] CardBundle 结构验证")
    for card_name in TEST_CARDS:
        names = extract_card_names(card_name, r)
        if not names:
            print(f"  FAIL: 无法提取卡名 '{card_name}'")
            errors.append(f"extract_card_names 失败: {card_name}")
            continue

        bundles = r.search_cards_by_name(names)
        if not bundles:
            print(f"  FAIL: 无法匹配卡名 '{card_name}'")
            errors.append(f"search_cards_by_name 失败: {card_name}")
            continue

        bundle = bundles[0]
        follower = bundle.get("follower")
        follower_evo = bundle.get("follower_evo")
        name = bundle["name_cn"]

        print(f"\n  {name}:")
        print(f"    Follower:     {'有' if follower else '无'} (card_no={follower.get('card_no','N/A') if follower else 'N/A'})")
        print(f"    FollowerEvo: {'有' if follower_evo else '无'} (card_no={follower_evo.get('card_no','N/A') if follower_evo else 'N/A'})")

        if "可可萝" in card_name:
            # 可可萝必须同时有 Follower 和 FollowerEvo
            assert follower is not None, f"FAIL: {card_name} 缺少 Follower"
            assert follower_evo is not None, f"FAIL: {card_name} 缺少 FollowerEvo"
            print(f"    PASS: 同时返回进化前/后")
        elif "佩可莉姆" in card_name:
            assert follower is not None, f"FAIL: {card_name} 缺少 Follower"
            assert follower_evo is not None, f"FAIL: {card_name} 缺少 FollowerEvo"
            print(f"    PASS: 同时返回进化前/后")
        elif "安＆古蕾娅" in card_name:
            # 安&古蕾娅仅 Follower（数据源中无 FollowerEvo）
            assert follower is not None, f"FAIL: {card_name} 缺少 Follower"
            print(f"    PASS: 仅进化前（数据源中无进化后，3条效果均在Follower上）")
        elif "灼热风暴" in card_name:
            # 法术卡：无 Follower，应有 Spell
            spell = bundle.get("spell")
            assert spell is not None, f"FAIL: {card_name} 缺少 Spell"
            print(f"    PASS: 法术卡正确识别")
        elif "灭剑焰龙" in card_name:
            assert follower_evo is not None, f"FAIL: {card_name} 缺少 FollowerEvo"
            print(f"    PASS: 仅进化后 Token 卡正确")

    # ===== 测试2: 可可萝效果完整性 =====
    print("\n" + "=" * 60)
    print("[测试2] 可可萝效果完整性验证")
    names = extract_card_names("可可萝", r)
    bundles = r.search_cards_by_name(names)
    bundle = bundles[0]

    follower_text = bundle["follower"]["text_preview"]
    follower_evo_text = bundle["follower_evo"]["text_preview"]

    checks = {
        "Follower 入场曲": "入场曲" in follower_text,
        "Follower UB": "UB" in follower_text,
        "Follower 召唤/爱梅斯": "爱梅斯" in follower_text,
        "FollowerEvo 进化时": "进化时" in follower_evo_text,
        "FollowerEvo 超进化": "超进化" in follower_evo_text,
        "FollowerEvo 抽卡": "抽取" in follower_evo_text or "引く" in follower_evo_text,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {check}")

    assert all_pass, "可可萝效果完整性检查失败"
    print("  PASS: 全部效果字段完整")

    # ===== 测试3: 安&古蕾娅效果完整性 =====
    print("\n" + "=" * 60)
    print("[测试3] 安&古蕾娅效果完整性验证（仅Follower，数据源无进化后）")
    # 用括号包裹全名来提取
    q = "【马纳历亚的亲爱挚友·安＆古蕾娅】"
    names = extract_card_names(q, r)
    bundles = r.search_cards_by_name(names)
    bundle = bundles[0]

    follower_text = bundle["follower"]["text_preview"]
    # 安&古蕾娅没有 FollowerEvo
    follower_evo = bundle.get("follower_evo")
    assert follower_evo is None, "安&古蕾娅不应有 FollowerEvo"

    checks = {
        "Follower 疾驰": "疾驰" in follower_text,
        "Follower 结束阶段": "结束阶段" in follower_text,
        "Follower 安的巨大英灵": "安的巨大英灵" in follower_text,
        "Follower 入场曲": "入场曲" in follower_text,
        "Follower 舍弃2张": "舍弃" in follower_text and "2张" in follower_text,
        "Follower 5点伤害": "5点伤害" in follower_text,
        "Follower 抽取2张": "抽取2张" in follower_text or "抽取2张卡" in follower_text,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {check}")
        if not passed:
            # 显示实际文本帮助调试
            print(f"      实际文本片段: ...{follower_text[max(0, follower_text.find(check.split()[-1])-30):follower_text.find(check.split()[-1])+80]}...")

    assert all_pass, "安&古蕾娅效果完整性检查失败"

    # ===== 测试4: 状态感知测试 =====
    print("\n" + "=" * 60)
    print("[测试4] 状态感知检测")
    assert _detect_card_question_focus("进化后有什么能力？") == "evo_first"
    assert _detect_card_question_focus("入场曲是什么？") == "base_first"
    assert _detect_card_question_focus("超进化时会怎样？") == "evo_first"
    assert _detect_card_question_focus("UB怎么触发？") == "base_first"
    assert _detect_card_question_focus("这张卡什么效果？") == "balanced"
    print("  PASS: 全部状态检测正确")

    # ===== 测试5: Prompt 组装验证 =====
    print("\n" + "=" * 60)
    print("[测试5] Prompt 组装验证（可可萝 入场曲问题）")
    q = "【可可萝】的入场曲是什么？"
    names = extract_card_names(q, r)
    bundles = r.search_cards_by_name(names)
    prompt = build_prompt(q, [], bundles)

    checks = {
        "包含进化前区块": "=== 进化前 (Follower) ===" in prompt,
        "包含进化后区块": "=== 进化后 (FollowerEvo) ===" in prompt,
        "进化前在进化后前面": prompt.find("进化前") < prompt.find("进化后"),
        "进化前包含入场曲": "入场曲" in prompt.split("=== 进化后")[0],
        "进化后包含进化时": "进化时" in prompt.split("=== 进化后")[1] if "=== 进化后" in prompt else False,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {check}")

    assert all_pass, "Prompt 组装验证失败"

    # ===== 测试6: Prompt 组装验证（进化后问题，顺序应颠倒）=====
    print("\n" + "=" * 60)
    print("[测试6] Prompt 组装验证（可可萝 进化后问题，应 evo_first）")
    q = "【可可萝】进化后会获得什么能力？"
    names = extract_card_names(q, r)
    bundles = r.search_cards_by_name(names)
    prompt = build_prompt(q, [], bundles)

    evo_pos = prompt.find("=== 进化后 (FollowerEvo) ===")
    base_pos = prompt.find("=== 进化前 (Follower) ===")
    assert evo_pos < base_pos, f"FAIL: 进化后应在进化前前面 (evo={evo_pos}, base={base_pos})"
    print("  PASS: 进化后优先展示（evo_first）")

    # ===== 测试7: 无括号卡名也能匹配 =====
    print("\n" + "=" * 60)
    print("[测试7] 无括号卡名子串匹配")
    q = "可以发动马纳历亚的亲爱挚友·安＆古蕾娅的入场曲么"
    names = extract_card_names(q, r)
    assert len(names) >= 1, f"FAIL: 无括号卡名提取失败，got {names}"
    found_target = any("安＆古蕾娅" in n for n in names)
    assert found_target, f"FAIL: 未匹配到安&古蕾娅，got {names}"
    print(f"  PASS: 提取到卡名 {names}")

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    if errors:
        print(f"测试失败！{len(errors)} 项错误:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("全部测试通过！CardBundle 重构成功。")

    return len(errors) == 0


if __name__ == "__main__":
    success = test_card_bundles()
    sys.exit(0 if success else 1)
