"""第 9 课：用 FastAPI 把 Agent 部署成 Web 服务

运行：uvicorn server:app --reload
然后打开 http://127.0.0.1:8000 使用聊天页面，或打开 http://127.0.0.1:8000/docs 查看接口文档

接口：
  GET  /                         聊天页面（static/index.html）
  GET  /health                   健康检查
  POST /chat                     非流式：等 agent 跑完一次性返回
  POST /chat/stream              流式（SSE）：逐 token 推送，同时推送工具调用和审批请求
  POST /chat/resume              提交审批决定，继续执行（SSE）
  GET  /threads/{thread_id}      查看会话历史

和命令行版 app.py 相比的变化：
  1. 全部用异步接口（ainvoke / astream），等待模型时不阻塞其他请求
  2. 会话存进 SQLite（AsyncSqliteSaver），服务重启后记忆不丢失
  3. 人工审批从"input() 阻塞等待"变成两段式：/chat/stream 推送审批请求后结束，
     前端展示审批界面，用户决定后调用 /chat/resume 继续
"""

import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import STREAM_MODES, build_agent, to_events
from common import ROOT, get_chat_model

if sys.version_info < (3, 11):
    # Python 3.10 及以下，contextvars 无法自动传递进异步任务，LangGraph 在 async 环境里调用 interrupt()
    # 会报 "Called get_config outside of a runnable context"，人工审批无法工作。
    raise RuntimeError("server.py 需要 Python 3.11 或更高版本，请用新版本 Python 重新创建虚拟环境")

DB_PATH = ROOT / "studymate.db"
STATIC_DIR = ROOT / "static"


# ---------------------------------------------------------------------------
# 请求体：Pydantic 自动校验，不合法的请求直接返回 422，不会进入业务代码
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, description="不传则创建新会话")


class Decision(BaseModel):
    type: Literal["approve", "edit", "reject"]
    message: str | None = None                   # reject 时告诉模型拒绝原因
    edited_action: dict[str, Any] | None = None  # edit 时的新参数：{"name": ..., "args": {...}}


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[Decision] = Field(min_length=1)


# ---------------------------------------------------------------------------
# 应用工厂：agent 的创建方式通过参数传入，测试时可以换成假模型
# ---------------------------------------------------------------------------

def default_agent_factory(checkpointer):
    from rag_advanced import build_default_retriever

    return build_agent(get_chat_model(), build_default_retriever(), checkpointer=checkpointer)


def create_app(agent_factory: Callable = default_agent_factory, db_path: Path = DB_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # lifespan：服务启动时执行 yield 之前的代码，关闭时执行之后的代码。
        # 数据库连接、模型、索引这类"重"资源在这里创建一次，所有请求共享。
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            app.state.agent = agent_factory(checkpointer)
            yield

    app = FastAPI(title="StudyMate API", description="LangChain 学习助手 Agent 服务", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        """非流式接口：简单，适合后端之间调用；缺点是用户要等全部生成完才能看到结果。"""
        thread_id = req.thread_id or new_thread_id()
        result = await request.app.state.agent.ainvoke(user_message(req.message), thread_config(thread_id))
        interrupts = result.get("__interrupt__")
        return {
            "thread_id": thread_id,
            "answer": result["messages"][-1].content if not interrupts else None,
            "approval_required": interrupts[0].value if interrupts else None,
        }

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
        thread_id = req.thread_id or new_thread_id()
        async for event in stream_agent(request.app.state.agent, user_message(req.message), thread_id):
            yield event

    async def require_pending_approval(req: ResumeRequest, request: Request) -> ResumeRequest:
        """依赖（Dependency）：在接口函数之前执行。

        流式接口一旦开始返回数据，状态码就已经发出（200）无法再改，在生成器里抛 HTTPException 没用。
        所以"这个会话有没有待审批的操作"这类前置校验要放进依赖，在流开始之前返回 409。
        """
        state = await request.app.state.agent.aget_state(thread_config(req.thread_id))
        if not state.interrupts:
            raise HTTPException(status_code=409, detail="该会话没有等待审批的操作")
        return req

    @app.post("/chat/resume", response_class=EventSourceResponse)
    async def chat_resume(
        req: Annotated[ResumeRequest, Depends(require_pending_approval)], request: Request
    ) -> AsyncIterator[ServerSentEvent]:
        decisions = [d.model_dump(exclude_none=True) for d in req.decisions]
        async for event in stream_agent(request.app.state.agent, Command(resume={"decisions": decisions}), req.thread_id):
            yield event

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str, request: Request):
        state = await request.app.state.agent.aget_state(thread_config(thread_id))
        if not state.values:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {
            "thread_id": thread_id,
            "messages": [
                {"role": m.type, "content": m.content}
                for m in state.values["messages"]
                if m.type in ("human", "ai") and m.content  # 只返回用户可见的对话，过滤掉工具调用细节
            ],
            "pending_approval": state.interrupts[0].value if state.interrupts else None,
        }

    return app


# ---------------------------------------------------------------------------
# SSE 流
# ---------------------------------------------------------------------------

def new_thread_id() -> str:
    return uuid.uuid4().hex


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def user_message(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


async def stream_agent(agent, payload, thread_id: str) -> AsyncIterator[ServerSentEvent]:
    """把 agent 的执行过程转成 SSE 事件流。

    每个事件在网络上的格式是：
        event: token
        data: {"text": "你好"}
        （空行）
    前端按 event 类型分别处理：token 拼接到回答里，tool_call 显示工具标签，approval_required 弹出审批框。
    """
    yield ServerSentEvent(event="start", data={"thread_id": thread_id})
    try:
        async for mode, data in agent.astream(payload, thread_config(thread_id), stream_mode=STREAM_MODES):
            for event in to_events(mode, data):
                kind = event.pop("type")
                if kind == "interrupt":
                    yield ServerSentEvent(event="approval_required", data=event["request"])
                else:
                    yield ServerSentEvent(event=kind, data=event)
    except Exception as e:
        # 流已经开始后就无法再返回 HTTP 500 了，只能用一个 error 事件通知前端
        yield ServerSentEvent(event="error", data={"message": f"{type(e).__name__}: {e}"})
    yield ServerSentEvent(event="done", data={"thread_id": thread_id})


app = create_app()
