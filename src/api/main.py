"""FastAPI 薄壳：把 BaseAgent 暴露成 HTTP 接口，供 web/index.html 使用。

设计约束（当前阶段）：
1. 单机、单用户、单会话：一份 agent.message 就够，不引入 sessions 字典；
2. 刷新页面要能恢复对话 -> 提供 GET /api/history；
3. "新建会话" = 清空服务端历史 -> POST /api/reset；
4. 不做流式：一次请求返回完整答案，前端直接渲染。

本文件只负责"定义 app"，不负责启动。
统一启动入口在项目根目录的 app.py：
    python app.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 保证从任何目录启动 uvicorn 都能 import src.*
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.qa_agent import BaseAgent      # noqa: E402
from src.log.logger import setup_logging   # noqa: E402

logger = setup_logging("api", "api.log")

# ---------------------------------------------------------------------------
# 全局状态：一个 agent = 一个 MCP 连接 = 一份对话历史
# ---------------------------------------------------------------------------
agent: BaseAgent | None = None
_lock = asyncio.Lock()        # 串行化对话，避免并发把 self.message 搅乱


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时连 MCP，退出时断开。MCP 是长连接，不能每个请求连一次。"""
    global agent
    agent = BaseAgent()
    await agent.init_mcp()
    tools = [t["function"]["name"] for t in agent.tools_schema]
    logger.info("API 启动，MCP 已连接，可用工具=%s", tools)
    try:
        yield
    finally:
        await agent.close_mcp()
        logger.info("API 退出")


app = FastAPI(title="三明学院问答 API", lifespan=lifespan)


class ChatIn(BaseModel):
    # 限长：挡住空问题和超长刷 token；500 字对一个校园问题足够
    question: str = Field(min_length=1, max_length=500, description="用户问题")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _need_agent() -> BaseAgent:
    if agent is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    return agent


def _public_history() -> list[dict]:
    """对外暴露的对话历史：只给 user/assistant 的纯文本。

    跳过开头两条 system 提示，也跳过带 tool_calls、content 为空的中间消息。
    """
    if agent is None:
        return []

    # 先拷一份：/api/chat 可能在线程里 append，reset 会截断列表，
    # 直接迭代原列表会读到半截状态（不是崩溃，但会出现错位的历史）
    messages = list(agent.message)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages[2:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


# ---------------------------------------------------------------------------
# 接口 1：历史（刷新页面时用来恢复对话）
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def history():
    return {"messages": _public_history()}


# ---------------------------------------------------------------------------
# 接口 2：新建会话（清空服务端历史，保留开头两条 system 提示）
# ---------------------------------------------------------------------------
@app.post("/api/reset")
async def reset():
    a = _need_agent()

    # 必须和 /api/chat 抢同一把锁：生成中清历史会让正在跑的那一轮留下
    # “没有对应 tool_calls 的 tool 消息”，之后每问必 400，只能重启。
    # 抢不到就直接拒绝，让前端提示用户等一会儿，而不是卡半天。
    try:
        await asyncio.wait_for(_lock.acquire(), timeout=3)
    except TimeoutError:
        raise HTTPException(status_code=409, detail="正在生成回答，请稍后再试")
    try:
        del a.message[2:]
    finally:
        _lock.release()

    logger.info("新建会话：历史已清空")
    return {"ok": True}


# ---------------------------------------------------------------------------
# 接口 3：问答（一次返回完整答案）
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(body: ChatIn):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    a = _need_agent()
    started = time.perf_counter()
    try:
        async with _lock:                  # 同一时刻只跑一轮对话
            answer = await a.chat(question)
    except Exception:
        # 上游超时 / 接口 4xx-5xx / 任何没预料到的错误，都转成一句人话，
        # 而不是让前端只看到 “500 Internal Server Error”
        logger.exception("chat 失败 q_len=%d", len(question))
        raise HTTPException(status_code=502, detail="模型服务暂时不可用，请稍后重试")

    logger.info(
        "chat q_len=%d a_chars=%d cost=%.0fms",
        len(question), len(answer or ""), (time.perf_counter() - started) * 1000,
    )
    return {"answer": answer or "（没有生成回答）"}


# ---------------------------------------------------------------------------
# 接口 4：健康检查
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    a = agent
    return {
        "ok": a is not None,
        "model": getattr(a, "model", None) if a else None,
        "tools": [t["function"]["name"] for t in a.tools_schema] if a else [],
        "messages": len(a.message) if a else 0,
    }


# ---------------------------------------------------------------------------
# 静态页面：同源托管 web/，免 CORS。
# 必须放在所有 /api 路由之后，否则会把接口一起兜住。
# ---------------------------------------------------------------------------
_web_dir = ROOT / "web"
if (_web_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")

