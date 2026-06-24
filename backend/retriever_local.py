"""
影之诗进化对决 AI裁判 —— 本地 FAISS 检索引擎
支持向量检索 + 卡名精确匹配二阶段策略

卡牌数据模型: CardBundle
  - 同一卡名（name_cn）按 card_type 分组
  - Follower 和 FollowerEvo 效果完全不同，必须分别保留
  - 同 card_type 内（如多版本稀有度）去重只保留最佳一条
"""
import json
import pickle
from pathlib import Path

import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "sve_faiss_index"
CHUNKS_DIR = PROJECT_ROOT / "rag_chunks"

# 每个 collection 检索条数权重
COLLECTION_TOP_K = {
    "rules": 5,
    "cards": 5,
    "qa": 8,        # QA 权重最高，优先参考
    "sections": 3,
}

# 需要分别保留的卡牌类型组（不同类型 = 不同卡牌状态，禁止去重）
# 凡是存在不同 card_type 配对的，两者都保留
SPLIT_TYPES = {
    ("Follower", "FollowerEvo"),
    ("FollowerAdv", "FollowerEvo"),  # 高阶随从也可能有进化后
    ("Spell", "SpellEvo"),
    ("Amulet", "AmuletEvo"),
}


def _build_card_meta(d):
    """从 card_chunk 的 dict 构建标准 meta 字典"""
    return {
        "id": d.get("id", d.get("card_no", "")),
        "source_type": "cards",
        "name_cn": d.get("name_cn", "").strip(),
        "card_no": d.get("card_no", ""),
        "title": d.get("name_cn", "").strip(),
        "text_preview": d.get("embedding_text", ""),
        "score": 1.0,
        "cost": d.get("cost", ""),
        "attack": d.get("attack", ""),
        "hp": d.get("life", d.get("hp", "")),
        "desc_cn": d.get("desc_cn", ""),
        "keywords": d.get("keywords", ""),
        "craft": d.get("craft", ""),
        "card_type": d.get("card_type", ""),
        "type": d.get("type", ""),
        "is_exact_card_match": True,
        "is_full_card_text": True,
    }


class LocalRetriever:
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = INDEX_DIR
        self.index_dir = Path(index_dir)
        self.indices = {}
        self.metas = {}
        self.encoder = None
        self.use_bge = False
        self._card_bundles = {}      # name_cn -> CardBundle
        self._card_names = set()     # 所有卡名，用于模糊匹配

        self._load()
        self._build_card_name_index()

    def _load(self):
        """加载 FAISS 索引和配置"""
        # 加载配置
        with open(self.index_dir / "config.pkl", "rb") as f:
            self.config = pickle.load(f)

        self.use_bge = self.config["embedding_model"].startswith("BGE")

        # 加载各 collection
        for name, meta in self.config["collections"].items():
            idx_path = self.index_dir / meta["index_file"]
            self.indices[name] = faiss.read_index(str(idx_path))

            meta_path = self.index_dir / meta["meta_file"]
            with open(meta_path, "rb") as f:
                self.metas[name] = pickle.load(f)

        # 加载 BGE 或 TF-IDF 编码器
        if self.use_bge:
            os_module = __import__("os")
            os_module.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer
            model_name = "BAAI/bge-m3" if "m3" in self.config["embedding_model"] else "BAAI/bge-small-zh-v1.5"
            self.encoder = SentenceTransformer(model_name)
        else:
            tfidf_path = self.index_dir / "tfidf_encoder.pkl"
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
            self.encoder = (data["vectorizer"], data["svd"])

    def _build_card_name_index(self):
        """
        构建 CardBundle 索引

        数据结构:
          self._card_bundles[name_cn] = {
            "name_cn": "可可萝",
            "follower": {...meta...},      # 进化前，None 表示没有
            "follower_evo": {...meta...},  # 进化后，None 表示没有
            "source_type": "cards",
            "is_exact_card_match": True,
            "is_full_card_text": True,
            "score": 1.0,
          }

        规则:
          - 同一 card_type 内去重（多版本稀有度 → 选最佳一条）
          - 不同 card_type 之间绝对不去重（Follower & FollowerEvo 效果完全不同）
        """
        card_chunks_path = CHUNKS_DIR / "card_chunks.jsonl"
        if not card_chunks_path.exists():
            return

        # 第一遍：收集所有数据，按 (name_cn, card_type) 分组
        from collections import defaultdict
        raw = defaultdict(lambda: defaultdict(list))  # name_cn -> card_type -> [metas]

        with open(card_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    name = d.get("name_cn", "").strip()
                    ct = d.get("card_type", "")
                    if not name or not ct:
                        continue
                    meta = _build_card_meta(d)
                    raw[name][ct].append(meta)
                    self._card_names.add(name)
                except json.JSONDecodeError:
                    continue

        # 第二遍：构建 CardBundle，同类型去重
        for name, type_groups in raw.items():
            bundle = {
                "name_cn": name,
                "follower": None,
                "follower_evo": None,
                "follower_adv": None,    # 高阶随从
                "spell": None,
                "spell_evo": None,
                "amulet": None,
                "amulet_evo": None,
                "source_type": "cards",
                "is_exact_card_match": True,
                "is_full_card_text": True,
                "score": 1.0,
            }
            for ct, metas in type_groups.items():
                best = self._dedup_same_type(metas)
                if best is None:
                    continue
                # 映射 card_type 到 bundle 字段
                key_map = {
                    "Follower": "follower",
                    "FollowerEvo": "follower_evo",
                    "FollowerAdv": "follower_adv",
                    "Spell": "spell",
                    "SpellEvo": "spell_evo",
                    "Amulet": "amulet",
                    "AmuletEvo": "amulet_evo",
                }
                bundle_key = key_map.get(ct)
                if bundle_key:
                    bundle[bundle_key] = best

            self._card_bundles[name] = bundle

    def _dedup_same_type(self, metas):
        """
        同 card_type 内去重：多个版本（不同稀有度/卡包）选最佳一条

        评分: desc_cn 有内容 + 3, attack != -1 + 1
        注意：此方法仅用于同类型去重，不同 card_type 由 CardBundle 保留
        """
        if not metas:
            return None
        if len(metas) == 1:
            return dict(metas[0])

        best = metas[0]
        best_score = 0
        for m in metas:
            score = 0
            if m.get("desc_cn"):
                score += 3
            if m.get("attack") and m.get("attack") != -1:
                score += 1
            if score > best_score:
                best_score = score
                best = m
        return dict(best)

    def encode_query(self, query):
        """将查询文本向量化"""
        if self.use_bge:
            emb = self.encoder.encode([query], normalize_embeddings=True)
            return emb.astype(np.float32)
        else:
            X = self.encoder[0].transform([query])
            emb = self.encoder[1].transform(X)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
            return emb.astype(np.float32)

    def search(self, query, top_k=None):
        """
        检索所有 collection，返回合并后的 TopK 结果

        返回:
          [{"source_type": "qa", "id": "...", "title": "...",
            "score": 0.85, "text_preview": "...", ...}, ...]
        """
        q_emb = self.encode_query(query)
        all_results = []

        for name, idx in self.indices.items():
            k = COLLECTION_TOP_K.get(name, 5)
            k = min(k, idx.ntotal)
            if k <= 0:
                continue

            scores, ids = idx.search(q_emb, k)
            for score, i in zip(scores[0], ids[0]):
                if i < 0 or i >= len(self.metas[name]):
                    continue
                meta = dict(self.metas[name][i])
                meta["score"] = float(score)
                all_results.append(meta)

        # 按分数降序
        all_results.sort(key=lambda x: x["score"], reverse=True)

        if top_k:
            all_results = all_results[:top_k]

        return all_results

    def search_qa_only(self, query, top_k=5):
        """仅检索 QA collection"""
        q_emb = self.encode_query(query)
        results = []

        for name in ["qa", "rules", "sections"]:
            if name not in self.indices:
                continue
            idx = self.indices[name]
            k = min(COLLECTION_TOP_K.get(name, 3), idx.ntotal)
            if k <= 0:
                continue
            scores, ids = idx.search(q_emb, k)
            for score, i in zip(scores[0], ids[0]):
                if i < 0 or i >= len(self.metas[name]):
                    continue
                meta = dict(self.metas[name][i])
                meta["score"] = float(score)
                results.append(meta)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_cards_by_name(self, card_names):
        """
        根据卡名列表精确匹配卡牌数据，返回 CardBundle 列表

        参数:
          card_names: ["灼热风暴", "可可萝", ...]

        返回:
          [CardBundle, ...]
          每个 CardBundle 包含 follower / follower_evo 等字段
          非随从卡（Spell/Amulet）只有对应字段
          精确匹配优先，score=1.0
        """
        results = []
        seen_names = set()

        for name in card_names:
            name = name.strip()
            if not name:
                continue

            # 1. 精确匹配
            if name in self._card_bundles:
                bundle = dict(self._card_bundles[name])
                bundle["is_exact_card_match"] = True
                bundle["score"] = 1.0
                results.append(bundle)
                seen_names.add(name)
                continue

            # 2. 模糊匹配：处理同音异字（天耀 vs 天曜）、前缀匹配等
            fuzzy_name = self._fuzzy_match_card_name(name)
            if fuzzy_name and fuzzy_name not in seen_names:
                bundle = dict(self._card_bundles[fuzzy_name])
                bundle["is_exact_card_match"] = False
                bundle["score"] = 0.9
                results.append(bundle)
                seen_names.add(fuzzy_name)
                continue

            # 3. 包含匹配（卡名含有关键词，用于简称匹配）
            for idx_name in self._card_bundles:
                if idx_name in seen_names:
                    continue
                if name in idx_name and name != idx_name:
                    bundle = dict(self._card_bundles[idx_name])
                    bundle["is_exact_card_match"] = False
                    bundle["score"] = 0.85
                    results.append(bundle)
                    seen_names.add(idx_name)
                    break

        return results

    def _fuzzy_match_card_name(self, name):
        """
        模糊匹配卡名，返回最佳匹配的 name_cn 或 None

        策略:
        1. 前2字前缀匹配 → 如果唯一，返回
        2. 首字+尾字+等长匹配 → 处理同音异字
        """
        if len(name) < 2:
            return None

        prefix = name[:2]
        matches = [n for n in self._card_names if n.startswith(prefix)]

        if len(matches) == 1:
            return matches[0]

        # 首字+尾字+等长匹配（天耀 → 天曜）
        if len(name) >= 3 and len(matches) > 1:
            first, last = name[0], name[-1]
            for n in matches:
                if n.endswith(last) and len(n) == len(name) and n != name:
                    return n

        return None


# 便捷函数
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = LocalRetriever()
    return _retriever
