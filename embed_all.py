"""
影之诗进化对决 AI裁判 —— 批量向量化脚本 (TF-IDF版)
将 rag_chunks/ 下的知识库全部向量化存入 ChromaDB
无需下载模型，纯离线运行

用法: 在项目根目录下运行
  cd "E:/桌面/作业/sveruler workbuddy"
  python embed_all.py
"""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import chromadb
from tqdm import tqdm

# ============ 配置 ============
BASE_DIR = Path.cwd()
RAG_DIR = BASE_DIR / "rag_chunks"
# 向量库存到不含中文的路径
VECTORDB_DIR = Path("D:/sve_vectordb")
EMBED_DIM = 384

FILES_CONFIG = [
    {
        "file": "rule_chunks.jsonl",
        "collection": "sve_rules",
        "text_field": "content",
        "meta_fields": ["id", "title", "rule_number", "section_title", "chapter"],
    },
    {
        "file": "card_chunks.jsonl",
        "collection": "sve_cards",
        "text_field": "embedding_text",
        "meta_fields": ["id", "card_no", "name_cn", "craft", "card_type", "cost"],
    },
    {
        "file": "qa_chunks_cn.jsonl",
        "collection": "sve_qa",
        "text_field": "embedding_text",
        "meta_fields": ["id", "qa_no", "Q_cn", "A_cn", "category"],
    },
    {
        "file": "section_chunks.jsonl",
        "collection": "sve_sections",
        "text_field": "content",
        "meta_fields": ["id", "title", "rule_number", "chapter"],
    },
]


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


def build_tfidf_encoder(texts, dim=EMBED_DIM):
    print("    构建 TF-IDF 词表 (char ngrams 1-3)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    n_features = X_tfidf.shape[1]
    print(f"    TF-IDF 特征维度: {n_features}, 耗时 {time.time()-t0:.1f}s")

    actual_dim = min(dim, n_features - 1, len(texts) - 1)
    actual_dim = max(actual_dim, 64)
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    print(f"    TruncatedSVD 降维到 {actual_dim} 维...")
    t0 = time.time()
    svd.fit(X_tfidf)
    print(f"    SVD 完成, 耗时 {time.time()-t0:.1f}s")
    explained = svd.explained_variance_ratio_.sum()
    print(f"    方差解释率: {explained:.1%}")
    return vectorizer, svd


def encode_batch(texts, vectorizer, svd):
    X_tfidf = vectorizer.transform(texts)
    X_dense = svd.transform(X_tfidf)
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_dense = X_dense / norms
    return X_dense


def main():
    print("=" * 60)
    print("  影之诗进化对决 AI裁判 —— 向量化脚本 (TF-IDF)")
    print("=" * 60)
    print(f"  工作目录: {BASE_DIR}")

    # [1] 收集全部文本
    print("\n[1/5] 收集全部文本，构建统一 TF-IDF 编码器...")
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
    print(f"    总计 {len(all_texts)} 条文本")

    vectorizer, svd = build_tfidf_encoder(all_texts)

    # 保存编码器
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    encoder_path = VECTORDB_DIR / "tfidf_encoder.pkl"
    with open(str(encoder_path), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
    print(f"    编码器已保存: {encoder_path}")

    # [2] 初始化 ChromaDB
    print(f"\n[2/5] 初始化 ChromaDB (持久化目录: {VECTORDB_DIR})")
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))

    # [3-4] 逐文件向量化
    for cfg in FILES_CONFIG:
        filepath = RAG_DIR / cfg["file"]
        if not filepath.exists():
            print(f"\n      !! 文件不存在，跳过: {cfg['file']}")
            continue

        col_name = cfg["collection"]
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        fsize_mb = filepath.stat().st_size / 1024 / 1024
        print(f"\n{'─' * 50}")
        print(f"[3] 读取: {cfg['file']} ({fsize_mb:.1f} MB)")

        records = load_jsonl(str(filepath))
        print(f"    共 {len(records)} 条记录 -> collection: {col_name}")

        BATCH_SIZE = 500
        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for start in tqdm(range(0, len(records), BATCH_SIZE),
                          desc=f"    Embedding {col_name}",
                          total=total_batches, ncols=80):
            batch = records[start:start + BATCH_SIZE]

            ids_list, texts_batch, metadatas = [], [], []
            for i, r in enumerate(batch):
                rid = r.get("id", f"{col_name}_{start + i}")
                text = safe_get(r, cfg["text_field"])
                if not text:
                    continue
                ids_list.append(rid)
                texts_batch.append(text)
                meta = {}
                for k in cfg["meta_fields"]:
                    val = safe_get(r, k)
                    if isinstance(val, str) and len(val) > 500:
                        val = val[:500]
                    meta[k] = str(val) if val else ""
                metadatas.append(meta)

            if not texts_batch:
                continue

            embeddings = encode_batch(texts_batch, vectorizer, svd)
            collection.add(
                ids=ids_list,
                embeddings=embeddings.tolist(),
                documents=texts_batch,
                metadatas=metadatas,
            )

        print(f"    OK {collection.count()} 条向量已入库")

    # [5] 验证
    print(f"\n{'─' * 50}")
    print("[5/5] 最终验证:")
    total = 0
    for cfg in FILES_CONFIG:
        try:
            col = client.get_collection(cfg["collection"])
            cnt = col.count()
            total += cnt
            print(f"    {cfg['collection']:20s} -> {cnt:>6} 条")
        except Exception:
            print(f"    {cfg['collection']:20s} -> 不存在")

    print(f"    {'─' * 30}")
    print(f"    总计: {total} 条向量")
    print(f"\n  OK 全部完成! 向量库位置: {VECTORDB_DIR}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
