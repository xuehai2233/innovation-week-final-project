"""
影之诗进化对决 AI裁判 —— 批量向量化脚本 (TF-IDF版)
将 rag_chunks/ 下的知识库全部向量化存入 ChromaDB
无需下载模型，纯离线运行

用法: 在项目根目录下运行
  cd "E:/桌面/作业/sveruler workbuddy"
  python embed_all.py
"""
# ↑ 以上为模块文档字符串（docstring），描述整个脚本的功能和用法，不会被解释器执行

# ═══════════════════════════════════════════════════════════════
# 一、导入模块
# ═══════════════════════════════════════════════════════════════

import json          # 标准库：解析 .jsonl 文件中每行的 JSON 对象（json.loads）
import os            # 标准库：操作系统相关功能（本脚本中路径辅助，主要用于兼容性保留）
import pickle        # 标准库：将训练好的 TfidfVectorizer / TruncatedSVD 对象序列化到磁盘
import time          # 标准库：记录各步骤耗时（time.time() 获取 Unix 时间戳）
from pathlib import Path  # 面向对象的路径操作：Path.cwd()、/ 拼接、.mkdir()、.stat()

import numpy as np   # 第三方库：数值计算，用于向量 L2 归一化（np.linalg.norm）
from sklearn.feature_extraction.text import TfidfVectorizer
# ↑ 文本 → TF-IDF 权重矩阵，使用字符级 n-gram 分析器
from sklearn.decomposition import TruncatedSVD
# ↑ 截断 SVD（奇异值分解），将高维稀疏 TF-IDF 降维到 384 维稠密向量
import chromadb       # 向量数据库客户端：创建/管理持久化向量集合，支持 HNSW + 余弦相似度
from tqdm import tqdm # 进度条库：批量处理时显示可视化进度

# ═══════════════════════════════════════════════════════════════
# 二、全局配置
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path.cwd()
# ↑ 获取当前工作目录（脚本运行时的目录），Path 对象支持 / 运算符拼接路径

RAG_DIR = BASE_DIR / "rag_chunks"
# ↑ 知识库切片（JSONL 文件）所在目录，/ 运算符得到子目录的绝对路径

VECTORDB_DIR = Path("D:/sve_vectordb")
# ↑ 向量数据库持久化目录，放在不含中文的 D 盘路径以避免 ChromaDB 底层 SQLite 兼容性问题

EMBED_DIM = 384
# ↑ 目标向量维度，TF-IDF → SVD 降维后的输出维度（检索精度与存储效率的平衡点）

FILES_CONFIG = [
    # ↓ 第一条配置：综合规则知识库
    {
        "file": "rule_chunks.jsonl",                      # JSONL 文件名
        "collection": "sve_rules",                        # ChromaDB 集合名称
        "text_field": "content",                          # 用于生成向量的文本字段
        "meta_fields": ["id", "title", "rule_number", "section_title", "chapter"],
        # ↑ 作为元数据保留的字段列表（便于检索时按规则编号、章节等过滤）
    },
    # ↓ 第二条配置：卡牌数据库
    {
        "file": "card_chunks.jsonl",
        "collection": "sve_cards",
        "text_field": "embedding_text",                   # 卡牌信息拼接后的专用向量化文本
        "meta_fields": ["id", "card_no", "name_cn", "craft", "card_type", "cost"],
    },
    # ↓ 第三条配置：QA 问答对
    {
        "file": "qa_chunks_cn.jsonl",
        "collection": "sve_qa",
        "text_field": "embedding_text",
        "meta_fields": ["id", "qa_no", "Q_cn", "A_cn", "category"],
    },
    # ↓ 第四条配置：规则章节切片
    {
        "file": "section_chunks.jsonl",
        "collection": "sve_sections",
        "text_field": "content",
        "meta_fields": ["id", "title", "rule_number", "chapter"],
    },
]
# ↑ 配置列表结束。每一个字典定义了：数据源文件 → 集合名 → 文本字段 → 元数据字段


# ═══════════════════════════════════════════════════════════════
# 三、工具函数
# ═══════════════════════════════════════════════════════════════

def load_jsonl(path):
    """
    读取并解析 JSONL 文件（每行一个独立 JSON 对象）
    返回解析成功的所有记录列表，跳过空行和解析失败的行（容错设计）
    """
    data = []
    # ↑ 初始化空列表，用于存储解析后的所有字典记录
    with open(path, "r", encoding="utf-8") as f:
        # ↑ 以只读模式、UTF-8 编码打开文件；with 语句确保文件使用完后自动关闭
        for line in f:
            # ↑ 逐行遍历文件内容（JSONL 格式每一行就是一个完整的 JSON 对象）
            line = line.strip()
            # ↑ 去除行首尾的空白字符（\n, \r, 空格, \t 等）
            if not line:
                # ↑ 如果去除空白后是空字符串（如文件末尾的空行）
                continue
                # ↑ 跳过空行，不做解析
            try:
                # ↑ 尝试解析 JSON 字符串
                data.append(json.loads(line))
                # ↑ 解析成功：转为 Python 字典/列表，追加到结果列表
            except json.JSONDecodeError:
                # ↑ 如果解析失败（格式损坏的行），捕获异常
                continue
                # ↑ 跳过该行，不中断整个文件读取（宁可丢失一行，不让导入全盘失败）
    return data
    # ↑ 返回所有解析成功的记录


def safe_get(obj, *keys, default=""):
    """
    安全的多级嵌套字典取值函数。
    逐级深入取嵌套字段，任一层不存在或非字典时返回默认值（而非抛异常）。
    示例：safe_get(r, "meta", "title") → r["meta"]["title"] 或 "" 若不存在
    """
    for k in keys:
        # ↑ 遍历传入的每一个键名，逐级深入
        if isinstance(obj, dict):
            # ↑ 只有当前层是字典时才能继续取值
            obj = obj.get(k, default)
            # ↑ .get() 安全取值：键存在返回对应值，不存在返回 default（不会 KeyError）
        else:
            # ↑ 当前对象已经不是字典（比如是字符串、None），无法继续取值
            return default
            # ↑ 提前返回默认值，避免 TypeError
    # 所有层级遍历完毕
    return obj if obj is not None else default
    # ↑ 返回最终值；若为 None 则降级为默认值，防止 None 污染下游处理


def build_tfidf_encoder(texts, dim=EMBED_DIM):
    """
    核心函数：用全部文本训练 TF-IDF 向量化器 + SVD 降维器。
    训练后返回 (vectorizer, svd) 二元组，供 encode_batch 和后续检索使用。
    参数：
        texts: 所有数据源的文本列表（列表长度 = 总记录数）
        dim:   目标降维维度，默认 384
    """
    print("    构建 TF-IDF 词表 (char ngrams 1-3)...")
    # ↑ 提示用户：开始构建字符级 n-gram（1-3）的词表
    t0 = time.time()
    # ↑ 记录开始时间戳，用于计算训练耗时

    vectorizer = TfidfVectorizer(
        # ↓ 构造 TF-IDF 向量化器，各项参数逐条说明：
        analyzer="char_wb",
        # ↑ 字符级分析器（word boundary）：只在词边界内提取 n-gram，
        #   不会跨越词边界产生无意义的字符组合，适合中日文混合文本。
        ngram_range=(1, 3),
        # ↑ 同时提取 1-gram（单字）、2-gram（双字）、3-gram（三字），捕获不同粒度的特征。
        max_features=30000,
        # ↑ 按词频排序，最多保留 30000 个特征词，防止维度爆炸。
        sublinear_tf=True,
        # ↑ 对词频取 1 + log(tf)，抑制高频词的过度影响，使分布更均匀。
    )
    X_tfidf = vectorizer.fit_transform(texts)
    # ↑ fit 学习 IDF 权重 + transform 转为 TF-IDF 稀疏矩阵（scipy.sparse.csr_matrix）
    #   .shape 为 (文本数, 特征数)，如 (15000, 28000)
    n_features = X_tfidf.shape[1]
    # ↑ 取列数 = 实际 TF-IDF 特征维度（≤ max_features=30000）
    print(f"    TF-IDF 特征维度: {n_features}, 耗时 {time.time()-t0:.1f}s")
    # ↑ 输出实际特征维度和训练耗时

    # ↓ 计算安全的 SVD 降维维度（三者取最小，防止数学错误）
    actual_dim = min(dim, n_features - 1, len(texts) - 1)
    # ↑ SVD 的 n_components 必须 ≤ 特征数-1 且 ≤ 样本数-1
    actual_dim = max(actual_dim, 64)
    # ↑ 保底不低于 64 维，避免降维过度导致语义信息严重丢失

    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    # ↑ 构造截断 SVD 降维器：
    #   n_components=actual_dim: 目标输出维度（384 或更小）
    #   random_state=42: 固定随机种子，保证每次运行结果可复现
    print(f"    TruncatedSVD 降维到 {actual_dim} 维...")
    # ↑ 提示降维目标维度

    t0 = time.time()
    # ↑ 重置计时器，开始记录 SVD 训练耗时
    svd.fit(X_tfidf)
    # ↑ 在 TF-IDF 矩阵上训练 SVD，学习高维→低维的线性变换矩阵
    #   本质上是对 TF-IDF 空间做 PCA 降维，提取最重要的语义主成分
    print(f"    SVD 完成, 耗时 {time.time()-t0:.1f}s")
    # ↑ 输出 SVD 训练耗时

    explained = svd.explained_variance_ratio_.sum()
    # ↑ 对所有成分的方差解释率求和，得到累计方差解释率
    #   值越接近 1.0（100%）说明降维后信息保留越好
    print(f"    方差解释率: {explained:.1%}")
    # ↑ 以百分比格式输出方差解释率（如 85.3%）

    return vectorizer, svd
    # ↑ 返回训练好的两个对象：TF-IDF 向量化器 + SVD 降维器


def encode_batch(texts, vectorizer, svd):
    """
    将一批文本编码为归一化稠密向量。
    流程：文本 → TF-IDF（稀疏） → SVD 降维（稠密） → L2 归一化
    返回 shape=(len(texts), EMBED_DIM) 的 numpy 数组
    """
    X_tfidf = vectorizer.transform(texts)
    # ↑ 用已训练的 TF-IDF 向量化器将新文本转为稀疏矩阵
    #   注意：只用 transform（不用 fit），因为词表/IDF 已固定
    X_dense = svd.transform(X_tfidf)
    # ↑ 用已训练的 SVD 将稀疏向量降维到目标维度（384 维），得到稠密矩阵

    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    # ↑ 按行（axis=1）计算每条向量的 L2 范数（欧氏长度）
    #   keepdims=True 保留维度形状（n,1）以支持广播除法
    norms[norms == 0] = 1.0
    # ↑ 将全零向量（空文本导致）的范数设为 1.0，防止除以零产生 NaN
    X_dense = X_dense / norms
    # ↑ L2 归一化：每条向量除以自身长度 → 变成单位向量（模为 1）
    #   归一化后内积等价于余弦相似度，与 ChromaDB 的 cosine 空间匹配
    return X_dense
    # ↑ 返回归一化后的稠密向量矩阵


# ═══════════════════════════════════════════════════════════════
# 四、主函数
# ═══════════════════════════════════════════════════════════════

def main():
    """
    主流程 —— 五步完成全量向量化：
    [1] 收集全部文本 → [2] 构建 TF-IDF + SVD 编码器 → [3] 初始化 ChromaDB
    → [4] 逐文件批量向量化入库 → [5] 验证
    """

    # ─────────── 启动横幅 ───────────
    print("=" * 60)
    # ↑ 输出 60 个等号组成的装饰线
    print("  影之诗进化对决 AI裁判 —— 向量化脚本 (TF-IDF)")
    # ↑ 脚本标题，明确标注使用 TF-IDF 方案
    print("=" * 60)
    # ↑ 装饰线闭合
    print(f"  工作目录: {BASE_DIR}")
    # ↑ 输出当前工作目录，方便用户确认路径正确

    # ─────────── [1/5] 收集全部文本 ───────────
    print("\n[1/5] 收集全部文本，构建统一 TF-IDF 编码器...")
    # ↑ 步骤标题，[1/5] 标识总进度
    all_texts = []
    # ↑ 初始化空列表，用于收集所有数据源的文本
    #   关键：所有文本集中训练同一套 TF-IDF 编码器，确保全局 IDF 一致
    for cfg in FILES_CONFIG:
        # ↑ 遍历 4 条文件配置（rules / cards / qa / sections）
        filepath = RAG_DIR / cfg["file"]
        # ↑ 拼接完整路径（如 rag_chunks/rule_chunks.jsonl）
        if not filepath.exists():
            # ↑ 文件不存在的容错检查
            continue
            # ↑ 跳过该数据源，不报错（优雅降级）
        records = load_jsonl(str(filepath))
        # ↑ 解析 JSONL 文件，str() 将 Path 对象转为字符串路径
        for r in records:
            # ↑ 遍历每条记录
            text = safe_get(r, cfg["text_field"])
            # ↑ 根据配置指定的文本字段（如 "content" / "embedding_text"）提取内容
            if text:
                # ↑ 仅保留非空文本
                all_texts.append(text)
                # ↑ 追加到全局文本列表
    print(f"    总计 {len(all_texts)} 条文本")
    # ↑ 输出收集到的文本总数

    # ─────────── 构建编码器并保存 ───────────
    vectorizer, svd = build_tfidf_encoder(all_texts)
    # ↑ 用全部文本训练 TF-IDF 向量化器 + SVD 降维器（得到全局统一的语义空间）

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    # ↑ 创建向量库目录，parents=True 递归创建父目录，exist_ok=True 已存在不报错
    encoder_path = VECTORDB_DIR / "tfidf_encoder.pkl"
    # ↑ 拼接编码器保存路径 → D:/sve_vectordb/tfidf_encoder.pkl
    with open(str(encoder_path), "wb") as f:
        # ↑ "wb" = 二进制写入模式，pickle 产生的是二进制数据
        pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
        # ↑ 将两个 sklearn 对象打包成字典，序列化写入文件
        #   检索时直接加载此文件即可得到训练好的编码器，无需重新训练
    print(f"    编码器已保存: {encoder_path}")
    # ↑ 确认持久化成功

    # ─────────── [2/5] 初始化 ChromaDB ───────────
    print(f"\n[2/5] 初始化 ChromaDB (持久化目录: {VECTORDB_DIR})")
    # ↑ 步骤标题
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    # ↑ 创建 ChromaDB 持久化客户端：
    #   PersistentClient → 数据存入磁盘（SQLite + HNSW 索引），重启不丢失
    #   path 指定存储目录，所有集合、向量、元数据都保存在此

    # ─────────── [3-4/5] 逐文件向量化入库 ───────────
    for cfg in FILES_CONFIG:
        # ↑ 第二轮遍历配置（第一轮仅收集文本，本轮是实际的向量化入库）
        filepath = RAG_DIR / cfg["file"]
        # ↑ 拼接完整路径
        if not filepath.exists():
            # ↑ 文件不存在检查
            print(f"\n      !! 文件不存在，跳过: {cfg['file']}")
            # ↑ 警告输出
            continue
            # ↑ 跳过

        col_name = cfg["collection"]
        # ↑ 获取集合名（sve_rules / sve_cards / sve_qa / sve_sections）
        try:
            client.delete_collection(col_name)
            # ↑ 先删除同名旧集合 → 保证每次运行都是全量重建（幂等性）
        except Exception:
            pass
            # ↑ 集合不存在时忽略异常（第一次运行或中途失败重跑的情况）

        collection = client.create_collection(
            # ↓ 创建新集合，参数说明：
            name=col_name,
            # ↑ 集合名称
            metadata={"hnsw:space": "cosine"},
            # ↑ 指定相似度计算方式为余弦距离（cosine distance）
            #   因为编码时已做 L2 归一化，向量内积 = 余弦相似度
        )

        fsize_mb = filepath.stat().st_size / 1024 / 1024
        # ↑ .stat().st_size 获取文件字节数，转换为 MB
        print(f"\n{'─' * 50}")
        # ↑ 输出 50 个 "─" 作为分隔线
        print(f"[3] 读取: {cfg['file']} ({fsize_mb:.1f} MB)")
        # ↑ 输出文件名和大小

        records = load_jsonl(str(filepath))
        # ↑ 再次解析 JSONL（之前收集文本时解析过但未保留）
        print(f"    共 {len(records)} 条记录 -> collection: {col_name}")
        # ↑ 输出记录数与目标集合

        BATCH_SIZE = 500
        # ↑ 每批处理 500 条记录：平衡内存占用和编码效率
        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        # ↑ 向上取整除法计算总批次数：(n + size - 1) // size 是标准写法
        #   例如 1200 条 → (1200 + 500 - 1) // 500 = 3 批

        for start in tqdm(range(0, len(records), BATCH_SIZE),
                          # ↑ 以 500 为步长生成起始索引 [0, 500, 1000, ...]
                          desc=f"    Embedding {col_name}",
                          # ↑ 进度条描述文字
                          total=total_batches,
                          # ↑ 总批次数（用于计算百分比）
                          ncols=80):
                          # ↑ 进度条固定宽度 80 列
            batch = records[start:start + BATCH_SIZE]
            # ↑ 切片取出当前批次的记录

            ids_list, texts_batch, metadatas = [], [], []
            # ↑ 初始化三个列表：ID 列表、文本列表、元数据列表
            for i, r in enumerate(batch):
                # ↑ i 是批内位置索引，r 是记录字典
                rid = r.get("id", f"{col_name}_{start + i}")
                # ↑ 获取记录 ID，不存在则用 {集合名}_{全局位置} 自动生成唯一 ID
                text = safe_get(r, cfg["text_field"])
                # ↑ 提取用于向量化的文本
                if not text:
                    # ↑ 跳过空文本记录
                    continue
                    # ↑ 不产生孤立向量，保持 ID/文本/元数据一一对应
                ids_list.append(rid)
                # ↑ ID 入列
                texts_batch.append(text)
                # ↑ 文本入列

                meta = {}
                # ↑ 初始化当前记录的元数据字典
                for k in cfg["meta_fields"]:
                    # ↑ 遍历配置的元数据字段（如 id, title, rule_number 等）
                    val = safe_get(r, k)
                    # ↑ 从记录中提取字段值
                    if isinstance(val, str) and len(val) > 500:
                        # ↑ 检查是否为超长字符串（ChromaDB 元数据有长度限制）
                        val = val[:500]
                        # ↑ 截断到 500 字符以内，防止插入失败
                    meta[k] = str(val) if val else ""
                    # ↑ 转为字符串存入元数据，空值填 ""
                metadatas.append(meta)
                # ↑ 元数据字典入列

            if not texts_batch:
                # ↑ 如果当前批次所有记录都没有有效文本
                continue
                # ↑ 跳过该批次（不编码不插入）

            embeddings = encode_batch(texts_batch, vectorizer, svd)
            # ↑ 批量编码：TF-IDF → SVD → L2 归一化 → (batch_n, 384) 向量矩阵
            collection.add(
                # ↓ ChromaDB 批量插入：
                ids=ids_list,
                # ↑ 每条向量的唯一 ID
                embeddings=embeddings.tolist(),
                # ↑ 向量列表（.tolist() 将 numpy 数组转为 Python 原生列表）
                documents=texts_batch,
                # ↑ 原始文本（支持全文召回）
                metadatas=metadatas,
                # ↑ 元数据字典列表（支持过滤查询，如按章节/职业筛选）
            )

        print(f"    OK {collection.count()} 条向量已入库")
        # ↑ 输出当前集合入库总数（用 .count() 验证与实际记录数是否一致）

    # ─────────── [5/5] 最终验证 ───────────
    print(f"\n{'─' * 50}")
    # ↑ 输出分隔线
    print("[5/5] 最终验证:")
    # ↑ 步骤标题
    total = 0
    # ↑ 初始化总向量计数
    for cfg in FILES_CONFIG:
        # ↑ 逐一检查 4 个集合
        try:
            col = client.get_collection(cfg["collection"])
            # ↑ 从 ChromaDB 获取集合对象
            cnt = col.count()
            # ↑ 获取集合中的向量数量
            total += cnt
            # ↑ 累加到总数
            print(f"    {cfg['collection']:20s} -> {cnt:>6} 条")
            # ↑ 格式化输出：集合名左对齐 20 列 / 数量右对齐 6 列
        except Exception:
            # ↑ 集合可能不存在（如对应 JSONL 文件缺失时）
            print(f"    {cfg['collection']:20s} -> 不存在")
            # ↑ 输出不存在信息

    print(f"    {'─' * 30}")
    # ↑ 小分隔线
    print(f"    总计: {total} 条向量")
    # ↑ 输出所有集合的向量总数
    print(f"\n  OK 全部完成! 向量库位置: {VECTORDB_DIR}")
    # ↑ 完成提示 + 向量库磁盘路径


# ═══════════════════════════════════════════════════════════════
# 五、脚本入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ↑ Python 惯用入口守卫：
    #   直接执行本文件时 __name__ == "__main__" 为 True → 执行整个流程
    #   被 import 导入时 __name__ == "embed_all" → 不执行 main()，避免误触发
    start = time.time()
    # ↑ 记录脚本启动时刻
    main()
    # ↑ 调用主函数，执行全部五步向量化流程
    elapsed = time.time() - start
    # ↑ 计算总耗时（秒）
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    # ↑ 输出总耗时（秒和分钟两种单位）
