"""第 9 课 FastAPI 服务的测试：用 TestClient 直接调用接口，不需要启动服务器。"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import build_agent
from server import create_app
from tests.fakes import fake_vector_store, scripted, tool_call


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def make_client(tmp_path):
    """创建一个用假模型的服务。同一个测试里多次调用会共用同一个 SQLite 文件，可以模拟服务重启。"""
    retriever = fake_vector_store().as_retriever()

    def _make(*messages):
        factory = lambda checkpointer: build_agent(
            scripted(*messages), retriever, checkpointer=checkpointer, notes_dir=tmp_path / "notes"
        )
        return TestClient(create_app(factory, db_path=tmp_path / "test.db"))

    return _make


def test_health_index_and_validation(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "StudyMate" in client.get("/").text
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_non_stream(make_client):
    with make_client(AIMessage("你好！")) as client:
        body = client.post("/chat", json={"message": "hi"}).json()
    assert body["answer"] == "你好！"
    assert body["approval_required"] is None
    assert body["thread_id"]


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


def test_resume_without_pending_approval_is_rejected(make_client):
    with make_client(AIMessage("hi")) as client:
        thread_id = client.post("/chat", json={"message": "hi"}).json()["thread_id"]
        resp = client.post("/chat/resume", json={"thread_id": thread_id, "decisions": [{"type": "approve"}]})
    assert resp.status_code == 409


def test_unknown_thread_returns_404(make_client):
    with make_client() as client:
        assert client.get("/threads/nope").status_code == 404


def test_stream_reports_errors_as_events(make_client):
    with make_client() as client:  # 剧本为空：假模型第一次调用就会抛出 StopIteration 之类的异常
        events = parse_sse(client.post("/chat/stream", json={"message": "hi"}).text)
    kinds = [e for e, _ in events]
    assert "error" in kinds and kinds[-1] == "done"
