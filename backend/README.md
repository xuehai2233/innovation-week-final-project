# 影之诗进化对决 AI裁判系统

> 本地运行的 Shadowverse EVOLVE 卡牌游戏规则裁判问答系统。
> 不依赖任何付费 API，全部在本地运行。

## 系统架构

```
前端 Web (app.html)
    ↓ POST /api/judge
FastAPI 后端 (main.py)
    ↓ 检索
FAISS 向量索引 (sve_faiss_index/)
    ↓ prompt + context
Ollama 本地大模型 (qwen2.5:7b)
    ↓ 生成回答
返回前端展示
```

## 前置条件

### 1. 安装 Ollama

下载安装: https://ollama.com/download/windows

安装完成后拉取模型:

```bash
# 推荐模型（7B，4GB显存可运行）
ollama pull qwen2.5:7b

# 备选模型
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull gemma3:4b
```

验证安装:

```bash
ollama list
# 应显示已拉取的模型
```

### 2. Python 环境

```bash
cd backend
pip install -r requirements.txt
```

## 启动步骤

### 第一步：构建 FAISS 向量索引（仅首次运行）

```bash
cd backend
python build_vector_db_local.py
```

这会:
1. 读取 `../rag_chunks/` 下的知识库（规则/卡牌/QA/章节）
2. 尝试加载 BGE-M3 embedding 模型
3. 如 BGE-M3 不可用，自动降级到 TF-IDF + SVD 方案
4. 生成 FAISS 索引到 `../sve_faiss_index/`

### 第二步：启动后端 API

```bash
cd backend
python main.py
```

服务启动在 http://localhost:8000

- API 文档: http://localhost:8000/docs
- 裁判接口: POST http://localhost:8000/api/judge
- 健康检查: GET http://localhost:8000/api/health

### 第三步：打开前端

浏览器打开 `backend/app.html`

## API 接口

### POST /api/judge

请求:
```json
{
  "question": "进化后攻击力怎么计算？"
}
```

返回:
```json
{
  "answer": "裁定：\n进化后的随从……\n\n依据：\n……\n\n解释：\n……\n\n结论：\n……",
  "sources": [
    {
      "id": "...",
      "source_type": "qa",
      "title": "关于进化后的攻击力计算",
      "score": 0.95,
      "text_preview": "..."
    }
  ]
}
```

### GET /api/health

检查后端和 Ollama 服务状态。

## Embedding 模型说明

系统支持三级降级:

| 优先级 | 模型 | 维度 | 说明 |
|--------|------|------|------|
| 1 | BAAI/bge-m3 | 1024 | 最佳中文语义效果，需下载约2.2GB |
| 2 | BAAI/bge-small-zh-v1.5 | 384 | 轻量中文模型，约100MB |
| 3 | TF-IDF + SVD | 384 | 纯离线，无需下载，保证可用 |

建议优先使用 BGE-M3，在网络受限时自动切换到 TF-IDF。

## LLM 模型推荐

| 模型 | 大小 | 显存需求 | 推荐场景 |
|------|------|----------|----------|
| qwen2.5:7b | ~4.7GB | 4-6GB | 首选，中文理解好 |
| qwen3:8b | ~5.5GB | 6-8GB | 最新版，效果更佳 |
| gemma3:4b | ~2.5GB | 3-4GB | 低配电脑首选 |
| llama3.1:8b | ~4.9GB | 4-6GB | 英文强，中文一般 |

## 项目结构

```
sveruler workbuddy/
├── rag_chunks/              # 知识库（已完成）
│   ├── rule_chunks.jsonl    # 规则 537条
│   ├── card_chunks.jsonl    # 卡牌 5937条
│   ├── qa_chunks_cn.jsonl   # 中文QA 7423条
│   └── section_chunks.jsonl # 章节 113条
├── sve_faiss_index/         # FAISS索引（build后生成）
├── backend/                 # 后端代码
│   ├── main.py              # FastAPI 入口
│   ├── build_vector_db_local.py  # 向量化脚本
│   ├── retriever_local.py   # FAISS检索
│   ├── local_llm.py         # Ollama调用
│   ├── rag_pipeline_local.py # RAG管线
│   ├── app.html             # 前端页面
│   ├── requirements.txt     # Python依赖
│   └── README.md            # 本文件
├── scripts/                 # 数据采集脚本（已完成）
└── sve_card_full_data.csv   # 卡牌原始数据
```

## 常见问题

**Q: Ollama 连接失败？**
确认 Ollama 在运行（任务栏应有 Ollama 图标），或执行 `ollama serve`

**Q: BGE-M3 下载太慢？**
脚本会自动降级到 TF-IDF 方案，效果也够用。

**Q: 模型回复太慢？**
CPU 推理确实慢，建议有 NVIDIA GPU 的电脑运行。或换用更小的模型如 gemma3:4b。

**Q: 前端跨域报错？**
后端已配置 CORS 允许所有来源，如仍有问题请确认访问的是 http://localhost:8000。
