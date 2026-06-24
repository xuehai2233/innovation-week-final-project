"""
影之诗进化对决 AI裁判 —— FastAPI 后端
启动: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

# 确保 backend 目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline_local import get_pipeline
from local_llm import LocalLLM

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="影之诗进化对决 AI裁判",
    description=" Shadowverse EVOLVE 卡牌裁判问答系统",
    version="1.0.0",
)

# CORS: 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class JudgeRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    id: str = ""
    source_type: str = ""
    title: str = ""
    score: float = 0.0
    text_preview: str = ""


class JudgeResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []


class HealthResponse(BaseModel):
    status: str
    deepseek: bool = False
    faiss_loaded: bool = False


# ==================== 延迟初始化 ====================
_pipeline = None
_llm = None

MODEL_NAME = "deepseek-v4-pro"
DEEPSEEK_API_KEY = "sk-ef513b10528c4f3aae51ce3e59bb6ae6"


def init_pipeline():
    global _pipeline, _llm
    if _pipeline is None:
        _llm = LocalLLM(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            reasoning_effort="medium",
        )
        _pipeline = get_pipeline(model=MODEL_NAME, llm=_llm)
    return _pipeline


# ==================== API 接口 ====================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    result = HealthResponse(status="ok")
    try:
        _l = LocalLLM(model=MODEL_NAME, api_key=DEEPSEEK_API_KEY)
        result.deepseek = _l.check_health()
    except Exception:
        pass

    try:
        from retriever_local import get_retriever
        r = get_retriever()
        result.faiss_loaded = len(r.indices) > 0
    except Exception:
        pass

    if not result.deepseek:
        result.status = "deepseek_unavailable"
    elif not result.faiss_loaded:
        result.status = "faiss_not_loaded"

    return result


@app.post("/api/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest):
    """
    裁判问答接口

    接收玩家提问，返回 AI 裁判的裁定回答和依据来源。
    """
    question = req.question.strip()
    if not question:
        return JudgeResponse(answer="请输入你的问题。", sources=[])

    pipeline = init_pipeline()

    try:
        result = pipeline.judge(question, return_sources=True)
        return JudgeResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
    except Exception as e:
        return JudgeResponse(
            answer=f"裁判系统内部错误: {str(e)}",
            sources=[],
        )


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import io, sys, uvicorn
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 60)
    print("  影之诗进化对决 AI裁判 - 后端服务")
    print("  API 文档: http://localhost:8000/docs")
    print("  裁判接口: POST http://localhost:8000/api/judge")
    print("=" * 60)

    # 预热
    try:
        print("\n  预加载 RAG 管线...")
        init_pipeline()
        print("  [OK] 管线就绪")
    except Exception as e:
        print(f"  [WARN] 管线预热失败: {e}")
        print("  服务仍可启动，首次请求时会自行初始化")

    uvicorn.run(app, host="0.0.0.0", port=8000)
