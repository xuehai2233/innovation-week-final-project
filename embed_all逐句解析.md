# embed_all.py 逐句代码解析

> **文件作用**：影之诗进化对决 AI 裁判系统的批量向量化脚本（TF-IDF 版），将 `rag_chunks/` 下的知识库全部向量化存入 ChromaDB，无需下载模型，纯离线运行。

---

## 一、文件头部文档字符串（第 1-9 行）

```python
"""
影之诗进化对决 AI裁判 —— 批量向量化脚本 (TF-IDF版)
将 rag_chunks/ 下的知识库全部向量化存入 ChromaDB
无需下载模型，纯离线运行

用法: 在项目根目录下运行
  cd "E:/桌面/作业/sveruler workbuddy"
  python embed_all.py
"""
```

| 行号 | 解释 |
|------|------|
| 1-9 | **模块文档字符串**。这是 Python 的模块级 docstring，描述整个脚本的功能和用法。三引号内的文字不会被解释器执行，但可通过 `help()` 或 `__doc__` 属性访问。作用是给开发者和使用者快速了解脚本用途和执行方式。 |

---

## 二、导入模块（第 10-20 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 10 | `import json` | 导入 Python 标准库 `json`，用于解析 `.jsonl` 文件中的 JSON 行数据（`json.loads()`）。 |
| 11 | `import os` | 导入 `os` 模块，提供操作系统相关功能（如路径操作）。本脚本中实际使用较少（主要用 `pathlib`），但为通用性而保留。 |
| 12 | `import pickle` | 导入 `pickle` 序列化库，用于将训练好的 `TfidfVectorizer` 和 `TruncatedSVD` 对象保存到磁盘（`pickle.dump()`），方便后续检索时直接加载而不需重新训练。 |
| 13 | `import time` | 导入 `time` 模块，用于记录各步骤耗时（`time.time()` 获取时间戳），帮助评估性能瓶颈。 |
| 14 | `from pathlib import Path` | 从 `pathlib` 导入 `Path` 类，提供面向对象的文件路径操作（`Path.cwd()`、`/` 拼接、`.mkdir()`、`.stat()` 等），替代传统的 `os.path`。 |
| 16 | `import numpy as np` | 导入 NumPy 科学计算库，别名 `np`。用于向量归一化（`np.linalg.norm`），是矩阵运算的基础。 |
| 17 | `from sklearn.feature_extraction.text import TfidfVectorizer` | 从 scikit-learn 导入 `TfidfVectorizer`，TF-IDF 文本向量化器。它将文本转为词频-逆文档频率权重矩阵，是传统 NLP 的经典特征提取方法。 |
| 18 | `from sklearn.decomposition import TruncatedSVD` | 导入截断 SVD（奇异值分解），用于对高维 TF-IDF 矩阵做降维处理，将数万维压缩到 384 维，同时保留最重要的语义信息。 |
| 19 | `import chromadb` | 导入 ChromaDB 向量数据库客户端，用于创建、管理和持久化向量集合。支持 HNSW 近似最近邻搜索和余弦相似度计算。 |
| 20 | `from tqdm import tqdm` | 导入 `tqdm` 进度条库，用于在批量处理时显示进度信息，提升用户体验和可观测性。 |

---

## 三、配置部分（第 22-54 行）

### 3.1 路径与维度配置（第 22-27 行）

```python
# ============ 配置 ============
BASE_DIR = Path.cwd()
RAG_DIR = BASE_DIR / "rag_chunks"
# 向量库存到不含中文的路径
VECTORDB_DIR = Path("D:/sve_vectordb")
EMBED_DIM = 384
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 22 | `# ============ 配置 ============` | 注释分隔线，标注以下为全局配置变量区。 |
| 23 | `BASE_DIR = Path.cwd()` | 获取**当前工作目录**（Current Working Directory）的 `Path` 对象。脚本运行时所在目录即项目根目录。 |
| 24 | `RAG_DIR = BASE_DIR / "rag_chunks"` | 使用 `/` 运算符拼接路径，得到 `rag_chunks` 子目录的绝对路径。这里是存放 `.jsonl` 知识库切片的目录。 |
| 25 | `# 向量库存到不含中文的路径` | 注释：ChromaDB 的底层 SQLite 对中文路径兼容性差，故选择不含中文的目录。 |
| 26 | `VECTORDB_DIR = Path("D:/sve_vectordb")` | 向量数据库持久化目录，存放在 D 盘根目录。ChromaDB 会将索引、元数据、向量数据以文件形式存储在此。 |
| 27 | `EMBED_DIM = 384` | 定义向量维度为 384。这是 TF-IDF → SVD 降维后的目标维度，在检索精度和存储效率之间取得平衡。 |

### 3.2 文件配置列表（第 29-54 行）

```python
FILES_CONFIG = [
    { "file": "rule_chunks.jsonl", ... },
    { "file": "card_chunks.jsonl", ... },
    { "file": "qa_chunks_cn.jsonl", ... },
    { "file": "section_chunks.jsonl", ... },
]
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 29 | `FILES_CONFIG = [` | 定义文件配置列表，每个元素是一个字典，描述一个需要向量化的数据源。 |
| 30-35 | 第一条配置 `rule_chunks.jsonl` | `file`: 文件名；`collection`: ChromaDB 集合名 `sve_rules`；`text_field`: 用于向量化的文本字段名 `content`；`meta_fields`: 作为元数据保留的字段（id, title, rule_number, section_title, chapter）。 |
| 36-41 | 第二条配置 `card_chunks.jsonl` | `collection` 为 `sve_cards`，`text_field` 为 `embedding_text`（这是卡牌信息拼接后的专用向量化文本），元数据包括卡牌编号、中文名、职业、类型、费用。 |
| 42-47 | 第三条配置 `qa_chunks_cn.jsonl` | `collection` 为 `sve_qa`，`text_field` 为 `embedding_text`，元数据包括 QA 编号、中文问答、分类。 |
| 48-53 | 第四条配置 `section_chunks.jsonl` | `collection` 为 `sve_sections`，`text_field` 为 `content`，元数据包括章节标题、规则编号。 |
| 54 | `]` | 配置列表结束。 |

---

## 四、工具函数（第 57-112 行）

### 4.1 load_jsonl()（第 57-68 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 57 | `def load_jsonl(path):` | 定义函数，接收 JSONL 文件路径作为参数。 |
| 58 | `data = []` | 初始化空列表，用于存储解析后的所有记录。 |
| 59 | `with open(path, "r", encoding="utf-8") as f:` | 以只读、UTF-8 编码打开文件。`with` 语句确保文件使用完毕后自动关闭，防止资源泄漏。 |
| 60 | `for line in f:` | 逐行遍历文件内容。JSONL 格式每行是一个独立的 JSON 对象。 |
| 61 | `line = line.strip()` | 去除行首尾的空白字符（换行符、空格、制表符等）。 |
| 62 | `if not line:` | 判断去除空白后是否为空字符串。 |
| 63 | `continue` | 跳过空行（如文件末尾的空白行），不作解析。 |
| 64 | `try:` | 开始一个异常处理块——尝试解析 JSON。 |
| 65 | `data.append(json.loads(line))` | 用 `json.loads()` 将字符串解析为 Python 字典/列表，并追加到 `data` 列表中。 |
| 66 | `except json.JSONDecodeError:` | 捕获 JSON 解析错误（如格式损坏的行）。 |
| 67 | `continue` | 跳过解析失败的行，不中断整个文件的读取。这是容错设计——宁可丢失一行脏数据，也不让整个导入失败。 |
| 68 | `return data` | 返回解析成功的所有记录列表。 |

### 4.2 safe_get()（第 71-77 行）

```python
def safe_get(obj, *keys, default=""):
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj if obj is not None else default
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 71 | `def safe_get(obj, *keys, default=""):` | 定义函数。`*keys` 是可变参数，支持多级嵌套取值（如 `safe_get(r, "meta", "title")`）。`default` 默认值为空字符串。 |
| 72 | `for k in keys:` | 遍历所有传入的键名，逐级深入。 |
| 73 | `if isinstance(obj, dict):` | 判断当前层是否还是字典。如果不是字典（如已是字符串或 None），则无法继续取值。 |
| 74 | `obj = obj.get(k, default)` | 从字典中安全取值，键不存在时返回 `default` 而非抛出 `KeyError`。 |
| 75 | `else:` | 当前对象不是字典时进入此分支。 |
| 76 | `return default` | 非字典类型无法继续取值，直接返回默认值。避免 `TypeError`。 |
| 77 | `return obj if obj is not None else default` | 所有层级遍历完毕后，返回最终值。若最终值为 `None`，则降级为默认值。防止 `None` 污染下游。 |

### 4.3 build_tfidf_encoder()（第 80-102 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 80 | `def build_tfidf_encoder(texts, dim=EMBED_DIM):` | 定义核心编码器构建函数。接收全部文本列表和目标维度，返回训练好的向量化器和降维器。 |
| 81 | `print(...)` | 输出提示信息，告知用户正在构建 TF-IDF 词表，使用字符级 1-3 gram。 |
| 82 | `t0 = time.time()` | 记录开始时间戳，用于后续计算耗时。 |
| 83-88 | `TfidfVectorizer(...)` | 构造 TF-IDF 向量化器实例：<br>• `analyzer="char_wb"`: 字符级分析器，仅在词边界内提取 n-gram（word boundary），适合中日文混合文本。<br>• `ngram_range=(1, 3)`: 同时使用 1-gram、2-gram、3-gram，捕获从单字到三字短语的特征。<br>• `max_features=30000`: 最多保留 30000 个特征词，按词频排序截断，防止维度爆炸。<br>• `sublinear_tf=True`: 对词频取 `1 + log(tf)`，抑制高频词的过度影响。 |
| 89 | `X_tfidf = vectorizer.fit_transform(texts)` | **训练并转换**：在所有文本上学习 IDF 权重，同时将文本转为 TF-IDF 稀疏矩阵。`fit_transform` 是 `fit` + `transform` 的组合。 |
| 90 | `n_features = X_tfidf.shape[1]` | 获取 TF-IDF 矩阵的列数（特征维度）。`shape[0]` 是文本数，`shape[1]` 是特征数。 |
| 91 | `print(...)` | 输出 TF-IDF 特征维度和训练耗时。 |
| 93 | `actual_dim = min(dim, n_features - 1, len(texts) - 1)` | 计算实际可用的 SVD 降维维度，取三者最小值，防止 `n_components` 超过样本数或特征数导致的数学错误。 |
| 94 | `actual_dim = max(actual_dim, 64)` | 保底维度至少为 64 维，避免降维过度导致信息损失严重。 |
| 95 | `svd = TruncatedSVD(n_components=actual_dim, random_state=42)` | 构造截断 SVD 降维器：<br>• `n_components=actual_dim`: 目标输出维度。<br>• `random_state=42`: 固定随机种子，保证每次运行结果可复现。 |
| 96 | `print(...)` | 输出降维目标维度信息。 |
| 97 | `t0 = time.time()` | 重置计时器，开始记录 SVD 训练耗时。 |
| 98 | `svd.fit(X_tfidf)` | 在 TF-IDF 矩阵上训练 SVD，学习从高维到低维的线性变换矩阵。SVD 相当于对 TF-IDF 矩阵做主成分分析，提取最重要的语义方向。 |
| 99 | `print(...)` | 输出 SVD 完成耗时。 |
| 100 | `explained = svd.explained_variance_ratio_.sum()` | 对所有成分的方差解释率求和，得到累计方差解释率。越接近 100% 说明信息保留越好。 |
| 101 | `print(f"    方差解释率: {explained:.1%}")` | 以百分比格式输出方差解释率（如 `85.3%`）。 |
| 102 | `return vectorizer, svd` | 返回训练好的 TF-IDF 向量化器和 SVD 降维器，供编码和保存使用。 |

### 4.4 encode_batch()（第 105-111 行）

```python
def encode_batch(texts, vectorizer, svd):
    X_tfidf = vectorizer.transform(texts)
    X_dense = svd.transform(X_tfidf)
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_dense = X_dense / norms
    return X_dense
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 105 | `def encode_batch(texts, vectorizer, svd):` | 定义编码函数，接收文本列表和训练好的编码器，返回归一化后的稠密向量矩阵。 |
| 106 | `X_tfidf = vectorizer.transform(texts)` | 用已训练的 TF-IDF 向量化器将文本转为稀疏矩阵。注意这里只用 `transform`（不用 `fit`），因为词表已固定。 |
| 107 | `X_dense = svd.transform(X_tfidf)` | 用已训练的 SVD 将稀疏 TF-IDF 向量降维到目标维度（384 维），得到稠密矩阵。 |
| 108 | `norms = np.linalg.norm(X_dense, axis=1, keepdims=True)` | 计算每个向量的 L2 范数（欧氏长度）。`axis=1` 表示按行计算；`keepdims=True` 保留维度以便广播除法。 |
| 109 | `norms[norms == 0] = 1.0` | 将范数为 0 的向量（全零向量/空文本）的范数设为 1.0，避免除以零导致 NaN。 |
| 110 | `X_dense = X_dense / norms` | L2 归一化：每个向量除以自身长度，使其变为单位向量。这是使用余弦相似度的前提——归一化后内积等于余弦相似度。 |
| 111 | `return X_dense` | 返回归一化后的向量矩阵。 |

---

## 五、主函数 main()（第 114-224 行）

### 5.1 启动标志（第 114-118 行）

```python
def main():
    print("=" * 60)
    print("  影之诗进化对决 AI裁判 —— 向量化脚本 (TF-IDF)")
    print("=" * 60)
    print(f"  工作目录: {BASE_DIR}")
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 114 | `def main():` | 定义主函数，脚本的核心入口。 |
| 115 | `print("=" * 60)` | 输出 60 个等号组成的装饰线，作为启动横幅。 |
| 116 | `print("  影之诗进化对决 AI裁判 —— 向量化脚本 (TF-IDF)")` | 输出脚本标题，明确标注使用 TF-IDF 方案。 |
| 117 | `print("=" * 60)` | 输出装饰线闭合。 |
| 118 | `print(f"  工作目录: {BASE_DIR}")` | 输出当前工作目录，方便用户确认路径正确性。 |

### 5.2 步骤 [1/5]：收集全部文本（第 120-132 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 120 | `# [1] 收集全部文本` | 注释，标注步骤 1。 |
| 121 | `print(...)` | 输出步骤标题 `[1/5]`，告知用户当前进度。 |
| 122 | `all_texts = []` | 初始化空列表，用于收集所有数据源的文本内容。**关键**：所有文本集中用于训练同一个 TF-IDF 编码器，确保全局 IDF 一致。 |
| 123 | `for cfg in FILES_CONFIG:` | 遍历 4 条文件配置。 |
| 124 | `filepath = RAG_DIR / cfg["file"]` | 拼接完整文件路径（`rag_chunks/rule_chunks.jsonl` 等）。 |
| 125 | `if not filepath.exists():` | 检查文件是否存在。 |
| 126 | `continue` | 文件不存在则跳过该数据源（不报错，优雅降级）。 |
| 127 | `records = load_jsonl(str(filepath))` | 调用 `load_jsonl` 解析文件，`str()` 将 `Path` 对象转为字符串路径。 |
| 128 | `for r in records:` | 遍历每条记录。 |
| 129 | `text = safe_get(r, cfg["text_field"])` | 根据配置指定的文本字段提取内容。 |
| 130 | `if text:` | 判断文本是否非空。 |
| 131 | `all_texts.append(text)` | 将非空文本追加到全局列表。 |
| 132 | `print(f"    总计 {len(all_texts)} 条文本")` | 输出收集到的文本总数。 |

### 5.3 构建编码器并保存（第 134-141 行）

```python
    vectorizer, svd = build_tfidf_encoder(all_texts)

    # 保存编码器
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    encoder_path = VECTORDB_DIR / "tfidf_encoder.pkl"
    with open(str(encoder_path), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
    print(f"    编码器已保存: {encoder_path}")
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 134 | `vectorizer, svd = build_tfidf_encoder(all_texts)` | 调用核心函数，用全部文本训练 TF-IDF 向量化器和 SVD 降维器。两个对象是整个向量化流程的核心。 |
| 136 | `# 保存编码器` | 注释。 |
| 137 | `VECTORDB_DIR.mkdir(parents=True, exist_ok=True)` | 创建向量库目录。`parents=True` 表示递归创建父目录；`exist_ok=True` 表示已存在时不报错。 |
| 138 | `encoder_path = VECTORDB_DIR / "tfidf_encoder.pkl"` | 拼接编码器保存路径：`D:/sve_vectordb/tfidf_encoder.pkl`。`.pkl` 是 pickle 序列化文件后缀。 |
| 139 | `with open(str(encoder_path), "wb") as f:` | 以二进制写入模式打开文件。`"wb"` 是必需的——pickle 产生的是二进制数据，不可用文本模式。 |
| 140 | `pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)` | 将两个 sklearn 对象打包成一个字典，序列化写入文件。这是 Python 原生的对象持久化方式。 |
| 141 | `print(...)` | 确认编码器保存成功。 |

### 5.4 步骤 [2/5]：初始化 ChromaDB（第 143-145 行）

```python
    # [2] 初始化 ChromaDB
    print(f"\n[2/5] 初始化 ChromaDB (持久化目录: {VECTORDB_DIR})")
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 143 | `# [2] 初始化 ChromaDB` | 注释，标注步骤 2。 |
| 144 | `print(...)` | 输出步骤标题，告知持久化目录位置。 |
| 145 | `client = chromadb.PersistentClient(path=str(VECTORDB_DIR))` | 创建 ChromaDB 持久化客户端。`PersistentClient` 将数据存储在磁盘上（基于 SQLite + HNSW 索引），重启后数据不会丢失。`path` 指定存储目录。 |

### 5.5 步骤 [3-4]：逐文件向量化入库（第 147-207 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 147 | `# [3-4] 逐文件向量化` | 注释，合并标注步骤 3 和 4。 |
| 148 | `for cfg in FILES_CONFIG:` | 第二轮遍历 4 条文件配置（第一轮仅收集文本，本轮才是真正的向量化入库）。 |
| 149 | `filepath = RAG_DIR / cfg["file"]` | 拼接完整路径。 |
| 150 | `if not filepath.exists():` | 检查文件是否存在。 |
| 151 | `print(f"\n      !! 文件不存在，跳过: {cfg['file']}")` | 文件不存在时输出警告并告知跳过。 |
| 152 | `continue` | 跳过该数据源，处理下一个。 |
| 154 | `col_name = cfg["collection"]` | 获取集合名称（如 `sve_rules`）。 |
| 155 | `try:` | 开始异常处理块。 |
| 156 | `client.delete_collection(col_name)` | 先删除已存在的同名集合。这样每次运行都是**全量重建**，而不是增量追加，保证数据的幂等性。 |
| 157 | `except Exception:` | 捕获所有异常（通常是集合不存在时抛出）。 |
| 158 | `pass` | 忽略异常——集合不存在时无需删除，直接进入创建。 |
| 160-163 | `collection = client.create_collection(...)` | 创建新集合：<br>• `name=col_name`: 集合名称。<br>• `metadata={"hnsw:space": "cosine"}`: 指定相似度计算方式为**余弦距离**。由于向量已 L2 归一化，内积等价于余弦相似度。 |
| 165 | `fsize_mb = filepath.stat().st_size / 1024 / 1024` | 获取文件大小（字节）并转换为 MB，`st_size` 返回文件的字节数。 |
| 166-167 | `print(...)` | 输出分隔线和文件名、大小的提示。 |
| 169 | `records = load_jsonl(str(filepath))` | 再次解析 JSONL（之前收集文本时已解析过，但未保存，此处重新读取）。 |
| 170 | `print(f"    共 {len(records)} 条记录 -> collection: {col_name}")` | 输出记录数量和目标集合名。 |
| 172 | `BATCH_SIZE = 500` | 定义批处理大小为 500。每 500 条记录批量编码和入库，平衡效率和内存占用。 |
| 173 | `total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE` | **向上取整除法**，计算总批次数。`(n + size - 1) // size` 是标准的向上取整写法。 |
| 175-178 | `for start in tqdm(...)` | 批量迭代。<br>• `range(0, len(records), BATCH_SIZE)`: 以 500 为步长生成起始索引。<br>• `tqdm`: 包裹迭代器显示进度条。<br>• `desc`: 进度条描述文字。<br>• `total`: 总批次数。<br>• `ncols=80`: 进度条宽度固定 80 列。 |
| 178 | `batch = records[start:start + BATCH_SIZE]` | 切片取出当前批次的记录。 |
| 180 | `ids_list, texts_batch, metadatas = [], [], []` | 初始化三个列表，分别存储 ID、文本、元数据。 |
| 181 | `for i, r in enumerate(batch):` | 遍历批次中每条记录，`i` 是批内索引，`r` 是记录本身。 |
| 182 | `rid = r.get("id", f"{col_name}_{start + i}")` | 获取记录 ID。如果记录中没有 `id` 字段，则用集合名加全局索引自动生成唯一 ID（`{collection}_{position}` 格式）。 |
| 183 | `text = safe_get(r, cfg["text_field"])` | 提取向量化文本。 |
| 184 | `if not text:` | 检查文本是否为空。 |
| 185 | `continue` | 跳过无文本的记录（不产生孤立向量）。 |
| 186 | `ids_list.append(rid)` | 将 ID 加入列表。 |
| 187 | `texts_batch.append(text)` | 将文本加入列表。 |
| 188 | `meta = {}` | 初始化元数据字典。 |
| 189 | `for k in cfg["meta_fields"]:` | 遍历配置中指定的元数据字段。 |
| 190 | `val = safe_get(r, k)` | 从记录中提取字段值。 |
| 191 | `if isinstance(val, str) and len(val) > 500:` | 检查字段值是否为超过 500 字符的字符串。ChromaDB 对元数据有长度限制，过长会导致插入失败。 |
| 192 | `val = val[:500]` | 截断到 500 字符以内。 |
| 193 | `meta[k] = str(val) if val else ""` | 将值转为字符串存入元数据字典，空值填充空字符串。 |
| 194 | `metadatas.append(meta)` | 将元数据字典加入列表。 |
| 196 | `if not texts_batch:` | 如果当前批次没有有效文本（全部被跳过）。 |
| 197 | `continue` | 跳过该批次（不调用编码器和插入）。 |
| 199 | `embeddings = encode_batch(texts_batch, vectorizer, svd)` | 批量编码：TF-IDF → SVD 降维 → L2 归一化，得到 `(batch_size, 384)` 的向量矩阵。 |
| 200-205 | `collection.add(...)` | 将向量批量插入 ChromaDB：<br>• `ids`: 唯一标识符列表。<br>• `embeddings`: 向量列表（`tolist()` 将 numpy 数组转为 Python 列表）。<br>• `documents`: 原始文本，支持全文召回。<br>• `metadatas`: 元数据字典列表，支持过滤查询。 |
| 207 | `print(f"    OK {collection.count()} 条向量已入库")` | 输出集合中实际入库的向量数量，用于验证。 |

### 5.6 步骤 [5/5]：最终验证（第 209-224 行）

```python
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
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 209 | `# [5] 验证` | 注释，标注步骤 5。 |
| 210-211 | `print(...)` | 输出分隔线和验证步骤标题。 |
| 212 | `total = 0` | 初始化总向量计数。 |
| 213 | `for cfg in FILES_CONFIG:` | 遍历所有配置，逐一检查。 |
| 214 | `try:` | 开始异常处理块——集合可能不存在。 |
| 215 | `col = client.get_collection(cfg["collection"])` | 从 ChromaDB 获取集合对象。 |
| 216 | `cnt = col.count()` | 获取集合中的向量数量。 |
| 217 | `total += cnt` | 累加到总数。 |
| 218 | `print(f"    {cfg['collection']:20s} -> {cnt:>6} 条")` | 格式化输出集合名（左对齐 20 列）和向量数（右对齐 6 列）。`>` 表示右对齐。 |
| 219 | `except Exception:` | 捕获获取失败的异常。 |
| 220 | `print(f"    {cfg['collection']:20s} -> 不存在")` | 输出该集合不存在的信息。 |
| 222-223 | `print(...)` | 输出装饰线和总计向量数。 |
| 224 | `print(f"\n  OK 全部完成! 向量库位置: {VECTORDB_DIR}")` | 输出完成信息，告知向量库的磁盘路径。 |

---

## 六、脚本入口（第 227-231 行）

```python
if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
```

| 行号 | 语句 | 解释 |
|------|------|------|
| 227 | `if __name__ == "__main__":` | Python 惯用入口守卫。当脚本直接执行时 `__name__` 为 `"__main__"`，条件成立；当作为模块导入时 `__name__` 为模块名，不执行。防止导入时意外触发全部向量化。 |
| 228 | `start = time.time()` | 记录脚本启动时间戳。 |
| 229 | `main()` | 调用主函数，执行全部向量化流程。 |
| 230 | `elapsed = time.time() - start` | 计算总耗时（秒）。 |
| 231 | `print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")` | 输出总耗时，同时以秒和分钟显示，方便评估性能。 |

---

## 七、整体流程总结

```
┌────────────────────────────────────────────────────────────┐
│                    embed_all.py 执行流程                      │
├────────────────────────────────────────────────────────────┤
│  [0] 加载配置（路径、文件列表、维度参数）                       │
│                          ↓                                   │
│  [1] 遍历所有 JSONL → 收集全部文本到 all_texts                │
│                          ↓                                   │
│  [2] 用全部文本训练 TF-IDF 向量化器 + SVD 降维器                │
│      保存为 tfidf_encoder.pkl                                │
│                          ↓                                   │
│  [3] 初始化 ChromaDB PersistentClient                        │
│                          ↓                                   │
│  [4] 逐文件遍历 → 删除旧集合 → 创建新集合                      │
│      批量编码（500条/批）→ L2归一化 → 插入 ChromaDB            │
│                          ↓                                   │
│  [5] 验证各集合向量数量 → 输出总结                             │
└────────────────────────────────────────────────────────────┘
```

| 设计要点 | 说明 |
|---------|------|
| **统一 IDF** | 所有文本共用一套 TF-IDF 词表，确保不同集合的向量在同一语义空间 |
| **全量重建** | 每次运行先删除旧集合再重新创建，保证幂等性 |
| **容错设计** | 空行跳过、解析失败跳过、文件缺失跳过、集合删除失败忽略——非致命错误不中断流程 |
| **L2 归一化** | 归一化后将内积转为余弦相似度，与 ChromaDB 的 `cosine` 空间匹配 |
| **批处理** | 500 条/批平衡内存和效率，避免一次性加载过多数据 |
| **离线可用** | 无需 GPU、无需下载模型，纯 CPU + scikit-learn 完成 |

---

> **关联文件**：
> - `backend/retriever_local.py`：加载 `tfidf_encoder.pkl`，对接 ChromaDB 进行检索
> - `backend/rag_pipeline_local.py`：组装 Prompt + 调用 LLM
> - `scripts/chunk_cards.py`、`scripts/scrape_qa.py`：生成 `.jsonl` 切片文件
