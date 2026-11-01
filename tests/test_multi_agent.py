"""第 8 课多 Agent 的测试。"""

import importlib

from langchain_core.messages import AIMessage, ToolMessage

from tests.fakes import fake_vector_store, scripted, tool_call

lesson = importlib.import_module("08_multi_agent")


def test_supervisor_delegates_to_subagents_with_isolated_context():
    # 主管和子 Agent 共用同一个假模型，剧本按实际调用顺序编排
    model = scripted(
        tool_call("analyze_jd", {"jd_text": "招 Agent 工程师"}),       # 主管 → JD 分析师
        AIMessage("核心技能：RAG、LangGraph"),                         # JD 分析师回答
        tool_call("research_topic", {"topic": "RAG 优化"}, "c2"),      # 主管 → 研究员
        tool_call("search_knowledge_base", {"query": "RAG 优化"}, "c3"),  # 研究员内部检索
        AIMessage("RAG 要点：混合检索、Rerank"),                        # 研究员回答
        tool_call("mock_interview", {"skills": "RAG"}, "c4"),           # 主管 → 面试官
        AIMessage("面试题 1：为什么要 Rerank？"),                       # 面试官回答
        AIMessage("面试准备报告：……"),                                 # 主管汇总
    )
    supervisor = lesson.build_supervisor(model, fake_vector_store().as_retriever())
    counter = lesson.CallCounter()
    result = supervisor.invoke({"messages": [{"role": "user", "content": "JD"}]}, {"callbacks": [counter]})

    assert result["messages"][-1].content == "面试准备报告：……"
    tool_results = {m.name: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_results["research_topic"] == "RAG 要点：混合检索、Rerank"
    # 上下文隔离：研究员内部的知识库检索不会出现在主管的消息历史里
    assert "search_knowledge_base" not in tool_results
    # 回调会传递到子 Agent：8 次模型调用全部被统计到
    assert counter.calls == 8


def test_map_reduce_workflow_researches_each_skill_in_parallel():
    model = scripted(
        tool_call("SkillList", {"skills": ["RAG", "LangGraph"]}),
        AIMessage("要点"), AIMessage("要点"),   # 两个并行分支各调用一次（顺序不确定，所以内容相同）
        AIMessage("最终报告"),
    )
    result = lesson.build_prep_workflow(model, fake_vector_store().as_retriever()).invoke({"jd": "JD"})

    assert result["skills"] == ["RAG", "LangGraph"]
    # reducer（operator.add）把两个分支的结果合并，而不是互相覆盖
    assert sorted(n.splitlines()[0] for n in result["notes"]) == ["## LangGraph", "## RAG"]
    assert result["report"] == "最终报告"
