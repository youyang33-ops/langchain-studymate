"""第 6 课：用 LangGraph 自己画图 —— "写作 → 审稿 → 修改" 循环工作流

运行：python 06_langgraph_workflow.py

create_agent 的循环是固定的（模型 ⇄ 工具）。当你需要确定的多步骤流程、
条件分支或循环时，就直接用 LangGraph 定义 State、Node、Edge。

本例流程：
    START → plan(列大纲) → write(写稿) → review(打分)
                              ↑               │
                              └── 分数不够 ───┤
                                              └── 分数够了 / 改稿次数用完 → END

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph  # 来自 langchain-ai/langgraph 仓库，不在 langchain 仓库
from pydantic import BaseModel, Field  # 第三方库 Pydantic，不属于 LangChain

from common import get_chat_model


# 【我写的】字段是我设计的，TypedDict 来自 Python 标准库
class WritingState(TypedDict, total=False):
    """整张图共享的状态。每个节点读取它，并返回要更新的字段。"""
    topic: str
    outline: str
    draft: str
    feedback: str
    score: int
    drafts_written: int


# 【我写的】字段是我设计的，BaseModel 和 Field 来自 Pydantic
class Review(BaseModel):
    score: int = Field(ge=1, le=10, description="综合评分，1-10 分")
    feedback: str = Field(description="具体、可执行的修改建议；如果已经很好则写'无需修改'")


# 【LangGraph 接口 + 我写的】LangGraph：StateGraph、add_node、add_edge、add_conditional_edges、compile；我写的：写作-审稿流程设计
def build_workflow(model, max_drafts: int = 3, pass_score: int = 8):
    """构建工作流。把 model 作为参数传入，方便测试时换成假模型。"""
    reviewer = model.with_structured_output(Review, method="function_calling")

    # ---- 节点：普通函数，输入 state，返回部分状态更新 ----

    # 【LangChain 接口 + 我写的】LangChain：model.invoke；我写的：提示词和状态更新
    def plan(state: WritingState) -> WritingState:
        outline = model.invoke(
            f"为一篇面向面试官的技术短文列 3 点大纲，主题：{state['topic']}。只输出大纲。"
        ).content
        return {"outline": outline, "drafts_written": 0}

    # 【LangChain 接口 + 我写的】LangChain：model.invoke；我写的：提示词和修改稿的分支逻辑
    def write(state: WritingState) -> WritingState:
        if state.get("feedback"):
            prompt = (
                f"根据审稿意见修改文章。\n审稿意见：{state['feedback']}\n\n原文：\n{state['draft']}\n\n"
                "只输出修改后的全文。"
            )
        else:
            prompt = f"按大纲写一篇 300 字左右的技术短文。\n主题：{state['topic']}\n大纲：\n{state['outline']}"
        return {"draft": model.invoke(prompt).content, "drafts_written": state["drafts_written"] + 1}

    # 【LangChain 接口 + 我写的】LangChain：with_structured_output 生成的 reviewer；我写的：提示词
    def review(state: WritingState) -> WritingState:
        result = reviewer.invoke(
            "你是严格的技术面试官，从准确性、条理性、是否有具体例子三个维度给文章打分。\n\n"
            f"主题：{state['topic']}\n文章：\n{state['draft']}"
        )
        return {"score": result.score, "feedback": result.feedback}

    # ---- 条件边：根据 state 决定下一步去哪 ----

    # 【我写的】判断条件是我写的，返回的 END 是 LangGraph 的常量
    def should_continue(state: WritingState) -> Literal["write", "__end__"]:
        if state["score"] >= pass_score or state["drafts_written"] >= max_drafts:
            return END
        return "write"

    # ---- 组装图 ----
    graph = StateGraph(WritingState)
    graph.add_node("plan", plan)
    graph.add_node("write", write)
    graph.add_node("review", review)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", should_continue, ["write", END])

    return graph.compile()


# 【LangGraph 接口 + 我写的】LangGraph：get_graph().draw_mermaid()、stream(stream_mode="updates")；我写的：打印进度
if __name__ == "__main__":
    workflow = build_workflow(get_chat_model(temperature=0.7))

    # 打印图结构（Mermaid 格式，可以粘贴到 https://mermaid.live 查看）
    print(workflow.get_graph().draw_mermaid())

    # stream_mode="updates"：每个节点执行完就输出它更新了什么，方便观察流程
    final_state = {}
    for update in workflow.stream({"topic": "LangChain 和 LangGraph 的区别与联系"}, stream_mode="updates"):
        for node, changes in update.items():
            final_state.update(changes)
            if node == "review":
                print(f"\n[review] 第 {final_state['drafts_written']} 稿得分 {changes['score']}：{changes['feedback']}")
            else:
                print(f"\n[{node}] 完成")

    print("\n========== 最终稿 ==========")
    print(final_state["draft"])
