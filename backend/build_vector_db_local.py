"""
影之诗进化对决 AI裁判 —— 本地 FAISS 向量索引构建
支持 BGE-M3 / bge-small-zh-v1.5 / TF-IDF 三级降级方案

用法:
  cd backend
  python build_vector_db_local.py
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

# 强制 UTF-8 输出，避免 GBK 编码问题
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from tqdm import tqdm

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "rag_chunks"
INDEX_DIR = PROJECT_ROOT / "sve_faiss_index"
EMBED_DIM = 1024  # BGE-M3 维度，降级时动态调整

FILES_CONFIG = [
    {
        "file": "rule_chunks.jsonl",
        "collection": "rules",
        "text_field": "content",
        "meta_fields": ["id", "title", "rule_number", "section_title", "chapter"],
    },
    {
        "file": "card_chunks.jsonl",
        "collection": "cards",
        "text_field": "embedding_text",
        "meta_fields": ["id", "card_no", "name_cn", "craft", "card_type", "cost"],
    },
    {
        "file": "qa_chunks_cn.jsonl",
        "collection": "qa",
        "text_field": "embedding_text",
        "meta_fields": ["id", "qa_no", "Q_cn", "A_cn", "category"],
    },
    {
        "file": "section_chunks.jsonl",
        "collection": "sections",
        "text_field": "content",
        "meta_fields": ["id", "title", "rule_number", "chapter"],
    },
]


# ===================== 工具函数 =====================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def safe_get(obj, *keys, default=""):
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj if obj is not None else default


# ===================== 方案1：BGE 模型 =====================

def try_load_bge():
    """尝试加载 BGE-M3，失败则降级到 bge-small-zh-v1.5，再失败返回 None"""
    # 设置 HF 镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [X] sentence-transformers not installed")
        return None

    for model_name in ["BAAI/bge-m3", "BAAI/bge-small-zh-v1.5"]:
        try:
            print(f"  Trying: {model_name} ...")
            # 清理不完整缓存
            import shutil
            cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
            if os.path.exists(cache_dir):
                snapshots = os.path.join(cache_dir, "snapshots")
                if os.path.exists(snapshots):
                    for s in os.listdir(snapshots):
                        sp = os.path.join(snapshots, s)
                        config_file = os.path.join(sp, "config_sentence_transformers.json")
                        if not os.path.exists(config_file):
                            print(f"  Cleaning partial cache: {cache_dir}")
                            shutil.rmtree(cache_dir, ignore_errors=True)
                            break

            model = SentenceTransformer(model_name)
            dim = model.get_sentence_embedding_dimension()
            print(f"  [OK] {model_name} loaded, dim={dim}")
            return model, dim
        except Exception as e:
            print(f"  [!] {model_name} failed: {e}")

    return None


# ===================== 方案2：TF-IDF 降级 =====================

def build_tfidf_encoder(texts, dim=384):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print("  构建 TF-IDF 词表 (char ngrams 1-3)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 3),
        max_features=30000, sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    n_features = X_tfidf.shape[1]
    print(f"  TF-IDF 特征维度: {n_features}, 耗时 {time.time()-t0:.1f}s")

    actual_dim = min(dim, n_features - 1, len(texts) - 1)
    actual_dim = max(actual_dim, 64)
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    print(f"  TruncatedSVD 降维到 {actual_dim} 维...")
    t0 = time.time()
    svd.fit(X_tfidf)
    print(f"  SVD 完成, 耗时 {time.time()-t0:.1f}s")
    print(f"  方差解释率: {svd.explained_variance_ratio_.sum():.1%}")
    return vectorizer, svd, actual_dim


# ===================== 主流程 =====================

def main():
    print("=" * 60)
    print("  影之诗进化对决 AI裁判 —— FAISS 向量索引构建")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")

    # ---- 收集全部文本 ----
    print("\n[1/5] 收集全部知识库文本...")
    all_texts = []
    for cfg in FILES_CONFIG:
        filepath = RAG_DIR / cfg["file"]
        if not filepath.exists():
            continue
        records = load_jsonl(str(filepath))
        for r in records:
            text = safe_get(r, cfg["text_field"])
            if text:
                all_texts.append(text)
    print(f"  总计 {len(all_texts)} 条文本")

    # ---- 加载 embedding 模型 ----
    print("\n[2/5] 加载 Embedding 模型...")
    encoder = None
    embed_dim = EMBED_DIM
    use_bge = False

    result = try_load_bge()
    if result:
        encoder, embed_dim = result
        use_bge = True
    else:
        print("\n  ⚠ BGE 模型全部不可用，降级到 TF-IDF 方案")
        tfidf_vectorizer, tfidf_svd, embed_dim = build_tfidf_encoder(all_texts)
        encoder = (tfidf_vectorizer, tfidf_svd)
        use_bge = False

    # ---- 初始化 FAISS ----
    print(f"\n[3/5] 初始化 FAISS (维度={embed_dim})...")
    import faiss
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    collections_meta = {}  # 存储每个 collection 的元数据

    # ---- 逐文件向量化 ----
    for cfg in FILES_CONFIG:
        filepath = RAG_DIR / cfg["file"]
        if not filepath.exists():
            print(f"\n  !! 文件不存在，跳过: {cfg['file']}")
            continue

        col_name = cfg["collection"]
        fsize_mb = filepath.stat().st_size / 1024 / 1024
        print(f"\n{'─' * 50}")
        print(f"[4] 处理: {cfg['file']} ({fsize_mb:.1f} MB) -> {col_name}")

        records = load_jsonl(str(filepath))
        print(f"  共 {len(records)} 条记录")

        texts_batch, metas_batch = [], []
        for r in records:
            text = safe_get(r, cfg["text_field"])
            if not text:
                continue
            texts_batch.append(text)
            meta = {"source_type": col_name}
            for k in cfg["meta_fields"]:
                val = safe_get(r, k)
                if isinstance(val, str) and len(val) > 500:
                    val = val[:500]
                meta[k] = str(val) if val else ""
            meta["text_preview"] = text[:200]
            metas_batch.append(meta)

        if not texts_batch:
            continue

        # embedding
        t0 = time.time()
        if use_bge:
            # BGE 模型
            emb = encoder.encode(
                texts_batch, show_progress_bar=True,
                normalize_embeddings=True, batch_size=64,
            )
        else:
            # TF-IDF 降级
            X_tfidf = encoder[0].transform(texts_batch)
            emb = encoder[1].transform(X_tfidf)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        print(f"  embedding 耗时: {time.time()-t0:.1f}s, shape={emb.shape}")

        # FAISS index
        idx = faiss.IndexFlatIP(emb.shape[1])  # Inner Product (cosine on normalized vecs)
        idx.add(emb.astype(np.float32))

        # 保存
        faiss.write_index(idx, str(INDEX_DIR / f"{col_name}.index"))
        with open(INDEX_DIR / f"{col_name}.meta.pkl", "wb") as f:
            pickle.dump(metas_batch, f)

        collections_meta[col_name] = {
            "count": len(metas_batch),
            "dim": emb.shape[1],
            "index_file": f"{col_name}.index",
            "meta_file": f"{col_name}.meta.pkl",
        }
        print(f"  [OK] {col_name}: {idx.ntotal} vectors stored in FAISS")

    # ---- 保存全局配置 ----
    config = {
        "embedding_model": "BGE-M3" if use_bge else "TF-IDF+SVD",
        "embedding_dim": embed_dim,
        "collections": collections_meta,
    }
    with open(INDEX_DIR / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    # ---- 保存 TF-IDF 编码器（如使用） ----
    if not use_bge:
        with open(INDEX_DIR / "tfidf_encoder.pkl", "wb") as f:
            pickle.dump({"vectorizer": encoder[0], "svd": encoder[1]}, f)

    # ---- 验证 ----
    print(f"\n{'─' * 50}")
    print("[5/5] 最终验证:")
    total = 0
    for name, meta in collections_meta.items():
        print(f"  {name:20s} -> {meta['count']:>6} 条, 维度={meta['dim']}")
        total += meta["count"]
    print(f"  {'─' * 30}")
    print(f"  总计: {total} 条向量")
    print(f"  模型: {config['embedding_model']}")
    print(f"\n  ✅ 全部完成! 索引位置: {INDEX_DIR}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
