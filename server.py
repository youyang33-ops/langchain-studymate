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

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request  # 第三方库 FastAPI，不属于 LangChain
from fastapi.responses import FileResponse  # 第三方库 FastAPI，不属于 LangChain
from fastapi.sse import EventSourceResponse, ServerSentEvent  # 第三方库 FastAPI，不属于 LangChain
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # 来自 langchain-ai/langgraph 仓库，不在 langchain 仓库
from langgraph.types import Command  # 来自 langchain-ai/langgraph 仓库，不在 langchain 仓库
from pydantic import BaseModel, Field  # 第三方库 Pydantic，不属于 LangChain

from app import STREAM_MODES, build_agent, to_events
from common import ROOT, get_chat_model

# 【我写的】检查 Python 版本，原因见下面的注释
if sys.version_info < (3, 11):
    # Python 3.10 及以下，contextvars 无法自动传递进异步任务，LangGraph 在 async 环境里调用 interrupt()
    # 会报 "Called get_config outside of a runnable context"，人工审批无法工作。
    raise RuntimeError("server.py 需要 Python 3.11 或更高版本，请用新版本 Python 重新创建虚拟环境")

DB_PATH = ROOT / "studymate.db"
STATIC_DIR = ROOT / "static"


# ---------------------------------------------------------------------------
# 请求体：Pydantic 自动校验，不合法的请求直接返回 422，不会进入业务代码
# ---------------------------------------------------------------------------

# 【我写的】请求体字段是我设计的，校验由 Pydantic 完成
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, description="不传则创建新会话")


# 【我写的】字段对应 LangChain 规定的审批决定格式
class Decision(BaseModel):
    type: Literal["approve", "edit", "reject"]
    message: str | None = None                   # reject 时告诉模型拒绝原因
    edited_action: dict[str, Any] | None = None  # edit 时的新参数：{"name": ..., "args": {...}}


# 【我写的】请求体字段是我设计的
class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[Decision] = Field(min_length=1)


# ---------------------------------------------------------------------------
# 应用工厂：agent 的创建方式通过参数传入，测试时可以换成假模型
# ---------------------------------------------------------------------------

# 【我写的】组装真实的模型和检索器
def default_agent_factory(checkpointer):
    from rag_advanced import build_default_retriever

    return build_agent(get_chat_model(), build_default_retriever(), checkpointer=checkpointer)


# 【LangGraph 接口 + 我写的】LangGraph：AsyncSqliteSaver；FastAPI：路由和依赖；我写的：接口设计
def create_app(agent_factory: Callable = default_agent_factory, db_path: Path = DB_PATH) -> FastAPI:
    # 【LangGraph 接口 + 我写的】LangGraph：AsyncSqliteSaver.from_conn_string；FastAPI：lifespan；我写的：资源初始化
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # lifespan：服务启动时执行 yield 之前的代码，关闭时执行之后的代码。
        # 数据库连接、模型、索引这类"重"资源在这里创建一次，所有请求共享。
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            app.state.agent = agent_factory(checkpointer)
            yield

    app = FastAPI(title="StudyMate API", description="LangChain 学习助手 Agent 服务", lifespan=lifespan)

    # 【我写的】FastAPI 路由，返回聊天页面
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    # 【我写的】FastAPI 路由，健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # 【LangGraph 接口 + 我写的】LangGraph：ainvoke、__interrupt__；我写的：返回内容的结构
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

    # 【我写的】FastAPI 的 SSE 路由，实际逻辑在 stream_agent
    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
        thread_id = req.thread_id or new_thread_id()
        async for event in stream_agent(request.app.state.agent, user_message(req.message), thread_id):
            yield event

    # 【LangGraph 接口 + 我写的】LangGraph：aget_state、interrupts；FastAPI：依赖；我写的：前置校验
    async def require_pending_approval(req: ResumeRequest, request: Request) -> ResumeRequest:
        """依赖（Dependency）：在接口函数之前执行。

        流式接口一旦开始返回数据，状态码就已经发出（200）无法再改，在生成器里抛 HTTPException 没用。
        所以"这个会话有没有待审批的操作"这类前置校验要放进依赖，在流开始之前返回 409。
        """
        state = await request.app.state.agent.aget_state(thread_config(req.thread_id))
        if not state.interrupts:
            raise HTTPException(status_code=409, detail="该会话没有等待审批的操作")
        return req

    # 【LangGraph 接口 + 我写的】LangGraph：Command(resume=...)；FastAPI：SSE 路由；我写的：接口逻辑
    @app.post("/chat/resume", response_class=EventSourceResponse)
    async def chat_resume(
        req: Annotated[ResumeRequest, Depends(require_pending_approval)], request: Request
    ) -> AsyncIterator[ServerSentEvent]:
        decisions = [d.model_dump(exclude_none=True) for d in req.decisions]
        async for event in stream_agent(request.app.state.agent, Command(resume={"decisions": decisions}), req.thread_id):
            yield event

    # 【LangGraph 接口 + 我写的】LangGraph：aget_state；我写的：过滤出用户可见的对话
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

# 【我写的】生成会话 id
def new_thread_id() -> str:
    return uuid.uuid4().hex


# 【LangGraph 用法】configurable.thread_id 配置格式
def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# 【我写的】小工具：拼出 agent 要求的消息字典
def user_message(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


# 【LangGraph 接口 + 我写的】LangGraph：astream；FastAPI：ServerSentEvent；我写的：事件设计和错误处理
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
