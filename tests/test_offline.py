"""离线测试：用"剧本式假模型"代替真实大模型，不需要 API Key 就能验证 agent 的流程逻辑。

运行：pytest -v

这也是一个值得学习的工程实践：agent 的"决策"来自模型，具有随机性，
但 agent 的"流程"（工具是否被正确执行、中断能否恢复、记忆是否保存）是确定的，
可以通过假模型按剧本返回预设消息来精确测试。

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

import importlib

import pytest  # 第三方库 pytest，不属于 LangChain
from langchain_core.messages import AIMessage, ToolMessage  # langchain 仓库 libs/core/langchain_core/messages/

from app import build_agent, run_turn
from common import load_documents, split_documents
from tests.fakes import fake_vector_store, scripted, tool_call
from tools import calculator

workflow_module = importlib.import_module("06_langgraph_workflow")


# 【我写的】pytest fixture：假检索器
@pytest.fixture(scope="module")
def fake_store():
    return fake_vector_store().as_retriever()


# 【我写的】小工具：生成会话配置
def thread(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


# ---------------------------------------------------------------------------
# 知识库与工具
# ---------------------------------------------------------------------------

# 【我写的】测试：切分后保留来源信息
def test_split_keeps_source_metadata():
    chunks = split_documents(load_documents())
    assert len(chunks) > len(load_documents())
    assert all(c.metadata["source"].endswith(".md") for c in chunks)
    assert max(len(c.page_content) for c in chunks) <= 500


# 【我写的】测试：计算器的正常计算
@pytest.mark.parametrize("expr,expected", [("(3+5)*2", "16"), ("10/4", "2.5"), ("-2**3", "-8")])
def test_calculator(expr, expected):
    assert calculator.invoke({"expression": expr}) == expected


# 【我写的】测试：计算器拒绝恶意或非法表达式
@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "1/0", "9**9**9"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    assert calculator.invoke({"expression": expr}).startswith("计算失败")


# ---------------------------------------------------------------------------
# Agent 流程
# ---------------------------------------------------------------------------

# 【我写的】测试：agent 调用检索工具后回答
def test_agent_uses_rag_tool_and_answers(fake_store, tmp_path, capsys):
    model = scripted(
        tool_call("search_knowledge_base", {"query": "checkpointer 作用"}),
        AIMessage("checkpointer 用于持久化状态。来源：04_langgraph.md"),
    )
    agent = build_agent(model, fake_store, notes_dir=tmp_path)
    config = thread("rag")
    run_turn(agent, "checkpointer 有什么用", config)

    # 流式过程中 artifact 是真正的 Document 对象，run_turn 据此打印命中的来源文件
    assert "命中" in capsys.readouterr().out
    # 注意：从 checkpointer 读回的 artifact 已被序列化成 dict，不再是 Document 对象
    messages = agent.get_state(config).values["messages"]
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert "[来源:" in tool_msg.content
    assert messages[-1].content.endswith("04_langgraph.md")


# 【我写的】测试：命令行打印工具轨迹，回答不重复
def test_run_turn_prints_answer_and_tool_trace(fake_store, tmp_path, capsys):
    model = scripted(tool_call("calculator", {"expression": "6*4"}), AIMessage("一共 24 小时"))
    agent = build_agent(model, fake_store, notes_dir=tmp_path)
    run_turn(agent, "6 个模块每个 4 小时，一共多少小时", thread("print"))

    out = capsys.readouterr().out
    assert "🔧 调用 calculator" in out
    assert "↳ calculator 返回" in out
    assert out.count("一共 24 小时") == 1  # 最终回答恰好打印一次，不重复


# 【我写的】测试：不同会话的记忆互相隔离
def test_memory_is_isolated_by_thread(fake_store, tmp_path):
    model = scripted(AIMessage("你好小林"), AIMessage("你叫小林"), AIMessage("我不知道"))
    agent = build_agent(model, fake_store, notes_dir=tmp_path)

    run_turn(agent, "我叫小林", thread("a"))
    run_turn(agent, "我叫什么", thread("a"))
    run_turn(agent, "我叫什么", thread("b"))

    assert len(agent.get_state(thread("a")).values["messages"]) == 4
    assert len(agent.get_state(thread("b")).values["messages"]) == 2


# 【我写的】测试：批准之前不会写入文件
def test_save_note_requires_approval(fake_store, tmp_path):
    model = scripted(
        tool_call("save_note", {"filename": "计划", "content": "# Day 1"}),
        AIMessage("已保存"),
    )
    agent = build_agent(model, fake_store, notes_dir=tmp_path)
    seen = []

    # 【我写的】测试用的审批函数
    def approve(request):
        seen.append(request)
        assert not (tmp_path / "计划.md").exists()  # 审批前绝不能已经写入文件
        return [{"type": "approve"}]

    run_turn(agent, "保存计划", thread("approve"), review=approve)
    assert len(seen) == 1
    assert (tmp_path / "计划.md").read_text(encoding="utf-8") == "# Day 1"


# 【我写的】测试：修改文件名和拒绝保存
def test_save_note_edit_and_reject(fake_store, tmp_path):
    model = scripted(
        tool_call("save_note", {"filename": "a", "content": "x"}, "c1"),
        AIMessage("已按新文件名保存"),
        tool_call("save_note", {"filename": "b", "content": "y"}, "c2"),
        AIMessage("好的，不保存"),
    )
    agent = build_agent(model, fake_store, notes_dir=tmp_path)
    config = thread("edit-reject")

    edit = lambda req: [{"type": "edit", "edited_action": {"name": "save_note", "args": {"filename": "renamed", "content": "x"}}}]
    run_turn(agent, "保存 a", config, review=edit)
    run_turn(agent, "保存 b", config, review=lambda req: [{"type": "reject", "message": "不要"}])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["renamed.md"]


# 【我写的】测试：文件名不能跳出笔记目录
def test_save_note_sanitizes_path_traversal(fake_store, tmp_path):
    model = scripted(tool_call("save_note", {"filename": "../../evil", "content": "x"}), AIMessage("ok"))
    agent = build_agent(model, fake_store, notes_dir=tmp_path / "notes")
    run_turn(agent, "保存", thread("evil"), review=lambda req: [{"type": "approve"}])

    assert [p.name for p in (tmp_path / "notes").iterdir()] == ["evil.md"]
    assert not (tmp_path / "evil.md").exists()


# ---------------------------------------------------------------------------
# LangGraph 工作流
# ---------------------------------------------------------------------------

# 【我写的】小工具：造一条审稿结果
def review(score: int, feedback: str) -> AIMessage:
    return tool_call("Review", {"score": score, "feedback": feedback})


# 【我写的】测试：审稿不通过时继续修改
def test_workflow_revises_until_pass():
    model = scripted(
        AIMessage("大纲"), AIMessage("初稿"), review(5, "缺少例子"), AIMessage("二稿"), review(9, "无需修改"),
    )
    final = workflow_module.build_workflow(model).invoke({"topic": "测试"})
    assert final["draft"] == "二稿"
    assert final["drafts_written"] == 2
    assert final["score"] == 9


# 【我写的】测试：达到稿数上限后停止
def test_workflow_stops_at_max_drafts():
    model = scripted(
        AIMessage("大纲"),
        AIMessage("稿1"), review(3, "差"),
        AIMessage("稿2"), review(4, "差"),
        AIMessage("稿3"), review(5, "差"),
    )
    final = workflow_module.build_workflow(model, max_drafts=3).invoke({"topic": "测试"})
    assert final["draft"] == "稿3"
    assert final["drafts_written"] == 3
