"""第 8 课：多 Agent —— 同一个需求，两种架构

运行：python 08_multi_agent.py

需求：输入一份招聘 JD，产出一份面试准备报告（技能分析 + 知识要点 + 模拟面试题）。

  方案 A  Supervisor 模式：主管 Agent 自己决定调用哪些子 Agent、调用几次
          → 灵活，能应对意外请求；但路径不确定、模型调用多
  方案 B  Map-Reduce 工作流：代码规定流程，每个技能并行调研
          → 可预测、可并行、便宜；但只能走预设流程

跑完对比两者的模型调用次数和耗时，体会"什么时候该让模型决策，什么时候该写死流程"。
"""

import operator
import time
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common import get_chat_model
from tools import make_search_tool

JD_TEXT = """
【某某智能】AI Agent 开发工程师（校招）— 杭州
职责：负责企业知识库问答系统和多 Agent 工作流的研发与优化。
要求：熟悉 Python；熟悉 LangChain / LangGraph；理解 RAG 原理，有检索效果优化经验者优先；
了解向量数据库；了解 MCP 协议者优先；有 FastAPI 服务开发经验。
"""


def last_text(result: dict) -> str:
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 方案 A：Supervisor —— 把子 Agent 包装成工具
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """你是求职辅导团队的主管，负责为用户生成面试准备报告。你手下有三位专家（以工具形式提供）：
- analyze_jd：分析招聘 JD，提炼核心技能
- research_topic：针对单个技术点检索知识库并整理要点（每个技术点单独调用一次）
- mock_interview：根据技能点生成模拟面试题

工作流程：先分析 JD，再挑选最重要的 2-3 个技术点分别调研，然后生成模拟面试题，
最后汇总成一份结构清晰的报告：岗位画像、重点知识、模拟面试题。不要遗漏任何一步。"""


def build_supervisor(model: BaseChatModel, retriever: BaseRetriever):
    # 每个子 Agent 有自己专属的提示词和工具，互不干扰
    jd_analyst = create_agent(
        model,
        system_prompt="你是招聘 JD 分析师。提炼：岗位核心职责、硬性技能（按重要性排序）、加分项、面试可能的考察重点。用简洁的要点列表输出。",
        name="jd_analyst",
    )
    researcher = create_agent(
        model,
        tools=[make_search_tool(retriever)],
        system_prompt="你是技术研究员。针对给定技术点，先检索知识库，再整理出面试需要掌握的 3-5 个要点，每点一两句话，末尾注明来源文件。",
        name="researcher",
    )
    coach = create_agent(
        model,
        system_prompt="你是资深面试官。根据给定的技能点，出 3 道由浅入深的面试题，每题附上参考答案要点。",
        name="coach",
    )

    # 关键：子 Agent 内部可能经历多轮检索和思考，但只把最终结果返回给主管。
    # 这就是"上下文隔离"，主管的上下文不会被子 Agent 的中间过程撑满。
    @tool
    def analyze_jd(jd_text: str) -> str:
        """分析招聘 JD，提炼核心职责、硬性技能、加分项和面试重点。收到 JD 时第一个调用。
        jd_text: 招聘 JD 原文"""
        return last_text(jd_analyst.invoke({"messages": [{"role": "user", "content": jd_text}]}))

    @tool
    def research_topic(topic: str) -> str:
        """针对单个技术点检索知识库，整理面试需要掌握的要点。每个技术点调用一次。
        topic: 具体的技术点，例如 RAG 检索优化、LangGraph 状态持久化"""
        return last_text(researcher.invoke({"messages": [{"role": "user", "content": topic}]}))

    @tool
    def mock_interview(skills: str) -> str:
        """根据技能点生成模拟面试题和参考答案要点。
        skills: 需要考察的技能点，用逗号分隔"""
        return last_text(coach.invoke({"messages": [{"role": "user", "content": skills}]}))

    return create_agent(
        model,
        tools=[analyze_jd, research_topic, mock_interview],
        system_prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )


# ---------------------------------------------------------------------------
# 方案 B：Map-Reduce 工作流 —— 用 Send 并行处理每个技能
# ---------------------------------------------------------------------------

class SkillList(BaseModel):
    skills: list[str] = Field(description="JD 中最重要的技术技能，按重要性排序，最多 3 个，每个是简短的技术名词")


class PrepState(TypedDict, total=False):
    jd: str
    skills: list[str]
    # Annotated + operator.add 是 reducer：多个并行分支各自返回 {"notes": [...]}，
    # LangGraph 会把它们拼接起来，而不是互相覆盖。
    notes: Annotated[list[str], operator.add]
    report: str


class SkillTask(TypedDict):
    """单个并行分支收到的输入。Send 可以给分支传入与主图不同的状态。"""
    skill: str


def build_prep_workflow(model: BaseChatModel, retriever: BaseRetriever):
    extractor = model.with_structured_output(SkillList, method="function_calling")

    def extract_skills(state: PrepState) -> PrepState:
        return {"skills": extractor.invoke(f"从下面的 JD 中提取核心技术技能：\n{state['jd']}").skills[:3]}

    def fan_out(state: PrepState) -> list[Send]:
        # 返回 Send 列表：为每个技能创建一个 research_skill 分支，这些分支在同一步中并行执行
        return [Send("research_skill", {"skill": s}) for s in state["skills"]]

    def research_skill(task: SkillTask) -> PrepState:
        docs = retriever.invoke(task["skill"])
        context = "\n\n".join(d.page_content for d in docs)
        note = model.invoke(
            f"根据资料，为面试整理「{task['skill']}」的 3 个关键要点和 1 道面试题。资料没有覆盖的部分可以用通用知识补充。\n\n资料：\n{context}"
        ).content
        return {"notes": [f"## {task['skill']}\n{note}"]}

    def write_report(state: PrepState) -> PrepState:
        report = model.invoke(
            "把下面各技能的调研笔记整合成一份面试准备报告，开头用一段话概括岗位画像，"
            f"保留每个技能的要点和面试题。\n\nJD：{state['jd']}\n\n" + "\n\n".join(state["notes"])
        ).content
        return {"report": report}

    graph = StateGraph(PrepState)
    graph.add_node("extract_skills", extract_skills)
    graph.add_node("research_skill", research_skill)
    graph.add_node("write_report", write_report)
    graph.add_edge(START, "extract_skills")
    graph.add_conditional_edges("extract_skills", fan_out, ["research_skill"])
    graph.add_edge("research_skill", "write_report")  # 所有并行分支完成后才会进入 write_report
    graph.add_edge("write_report", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 运行与对比
# ---------------------------------------------------------------------------

class CallCounter(BaseCallbackHandler):
    """回调（Callback）：统计模型被调用了多少次。LangSmith 的追踪也是基于回调机制实现的。"""

    def __init__(self):
        self.calls = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.calls += 1


def run_and_measure(name: str, runnable, payload: dict) -> dict:
    counter = CallCounter()
    start = time.perf_counter()
    result = runnable.invoke(payload, {"callbacks": [counter]})
    print(f"\n[{name}] 模型调用 {counter.calls} 次，耗时 {time.perf_counter() - start:.1f} 秒")
    return result


if __name__ == "__main__":
    from rag_advanced import build_default_retriever

    model = get_chat_model()
    retriever = build_default_retriever()

    print("=" * 20, "方案 A：Supervisor", "=" * 20)
    supervisor = build_supervisor(model, retriever)
    result_a = run_and_measure("Supervisor", supervisor, {"messages": [{"role": "user", "content": JD_TEXT}]})
    for msg in result_a["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"  主管调用 → {call['name']}({str(call['args'])[:60]})")
    print(last_text(result_a))

    print("\n" + "=" * 20, "方案 B：Map-Reduce 工作流", "=" * 20)
    workflow = build_prep_workflow(model, retriever)
    result_b = run_and_measure("工作流", workflow, {"jd": JD_TEXT})
    print(f"  提取的技能：{result_b['skills']}（并行调研）")
    print(result_b["report"])
