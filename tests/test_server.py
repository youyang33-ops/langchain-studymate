"""第 9 课 FastAPI 服务的测试：用 TestClient 直接调用接口，不需要启动服务器。

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

import json

import pytest  # 第三方库 pytest，不属于 LangChain
from fastapi.testclient import TestClient  # 第三方库 FastAPI，不属于 LangChain
from langchain_core.messages import AIMessage  # langchain 仓库 libs/core/langchain_core/messages/ai.py

from app import build_agent
from server import create_app
from tests.fakes import fake_vector_store, scripted, tool_call


# 【我写的】小工具：解析 SSE 响应
def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


# 【我写的】pytest fixture：创建用假模型的服务
@pytest.fixture
def make_client(tmp_path):
    """创建一个用假模型的服务。同一个测试里多次调用会共用同一个 SQLite 文件，可以模拟服务重启。"""
    retriever = fake_vector_store().as_retriever()

    # 【我写的】可重复调用，用来模拟服务重启
    def _make(*messages):
        factory = lambda checkpointer: build_agent(
            scripted(*messages), retriever, checkpointer=checkpointer, notes_dir=tmp_path / "notes"
        )
        return TestClient(create_app(factory, db_path=tmp_path / "test.db"))

    return _make


# 【我写的】测试：健康检查、首页、参数校验
def test_health_index_and_validation(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "StudyMate" in client.get("/").text
        assert client.post("/chat", json={"message": ""}).status_code == 422


# 【我写的】测试：非流式接口
def test_chat_non_stream(make_client):
    with make_client(AIMessage("你好！")) as client:
        body = client.post("/chat", json={"message": "hi"}).json()
    assert body["answer"] == "你好！"
    assert body["approval_required"] is None
    assert body["thread_id"]


# 【我写的】测试：流式接口的事件
def test_chat_stream_emits_tool_and_token_events(make_client):
    with make_client(tool_call("calculator", {"expression": "2*21"}), AIMessage("答案是 42")) as client:
        resp = client.post("/chat/stream", json={"message": "2 乘 21"})
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert ("tool_call", {"name": "calculator", "args": {"expression": "2*21"}}) in events
    assert any(e == "tool_result" and d["name"] == "calculator" for e, d in events)
    assert "".join(d["text"] for e, d in events if e == "token") == "答案是 42"


# 【我写的】测试：审批流程，以及服务重启后状态还在
def test_approval_flow_and_persistence_across_restart(make_client, tmp_path):
    script = (tool_call("save_note", {"filename": "计划", "content": "# Day 1"}), AIMessage("已保存"))

    with make_client(*script) as client:
        events = parse_sse(client.post("/chat/stream", json={"message": "保存计划"}).text)
        thread_id = events[0][1]["thread_id"]
        approval = next(d for e, d in events if e == "approval_required")
        assert approval["action_requests"][0]["name"] == "save_note"
        assert not (tmp_path / "notes" / "计划.md").exists()  # 审批前不能写入

    # 模拟服务重启：新建一个 app，共用同一个 SQLite 文件，等待审批的状态应该还在
    with make_client(*script[1:]) as client:
        assert client.get(f"/threads/{thread_id}").json()["pending_approval"] is not None
        events = parse_sse(client.post(
            "/chat/resume", json={"thread_id": thread_id, "decisions": [{"type": "approve"}]}
        ).text)
        assert "".join(d["text"] for e, d in events if e == "token") == "已保存"
        history = client.get(f"/threads/{thread_id}").json()

    assert (tmp_path / "notes" / "计划.md").read_text(encoding="utf-8") == "# Day 1"
    assert [m["role"] for m in history["messages"]] == ["human", "ai"]
    assert history["pending_approval"] is None


# 【我写的】测试：没有待审批操作时返回 409
def test_resume_without_pending_approval_is_rejected(make_client):
    with make_client(AIMessage("hi")) as client:
        thread_id = client.post("/chat", json={"message": "hi"}).json()["thread_id"]
        resp = client.post("/chat/resume", json={"thread_id": thread_id, "decisions": [{"type": "approve"}]})
    assert resp.status_code == 409


# 【我写的】测试：会话不存在时返回 404
def test_unknown_thread_returns_404(make_client):
    with make_client() as client:
        assert client.get("/threads/nope").status_code == 404


# 【我写的】测试：出错时发送 error 事件
def test_stream_reports_errors_as_events(make_client):
    with make_client() as client:  # 剧本为空：假模型第一次调用就会抛出 StopIteration 之类的异常
        events = parse_sse(client.post("/chat/stream", json={"message": "hi"}).text)
    kinds = [e for e, _ in events]
    assert "error" in kinds and kinds[-1] == "done"
