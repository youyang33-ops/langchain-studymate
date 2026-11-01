"""StudyMate —— LangChain 学习助手（综合项目）

运行：python app.py

需求清单（每一条对应前面某一课的知识点）：
  1. 基于本地知识库回答 LangChain 相关问题，并标注来源       → 第 4、7 课 RAG（Agentic RAG + 混合检索）
  2. 计算、查日期等确定性任务交给工具，不让模型猜              → 第 3 课 工具调用
  3. 能把学习计划、知识总结保存成笔记文件                      → 第 3 课 工具调用
  4. 保存文件前必须经过用户确认，用户可以批准/修改/拒绝        → 第 5 课 人工审批
  5. 支持多轮对话，输入 /new 开启新会话                        → 第 5 课 记忆
  6. 对话过长时自动摘要，防止超出上下文窗口                    → 第 5 课 中间件
  7. 限制单轮最多调用模型次数，防止死循环；工具出错不崩溃      → 第 5 课 中间件
  8. 流式输出回答，并实时展示 agent 调用了哪些工具             → 第 1 课 流式 + LangGraph stream
"""

import json
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from common import ROOT, get_chat_model
from tools import calculator, get_current_time, make_save_note_tool, make_search_tool

NOTES_DIR = ROOT / "notes"

SYSTEM_PROMPT = """你是 StudyMate，帮助用户学习 LangChain 并准备 Agent 开发岗位求职。

工作规则：
1. 回答 LangChain、LangGraph、RAG、Agent、面试相关问题时，必须先调用 search_knowledge_base 检索，
   基于检索结果回答，并在回答末尾用"来源：xxx.md"注明出处。
   如果知识库没有相关内容，要明确说"知识库未覆盖，以下是通用知识"，再谨慎回答。
2. 需要计算时调用 calculator，涉及今天日期时调用 get_current_time，不要心算或猜测。
3. 用户要求保存、记录、整理成笔记时，调用 save_note。如果用户拒绝保存，尊重决定，不要再次尝试。
4. 用中文回答，简洁、条理清晰。"""


class ToolErrorHandler(AgentMiddleware):
    """工具抛异常时，把错误信息作为工具结果返回给模型，让它向用户说明，而不是让程序崩溃。

    为什么不用第 5 课的 @wrap_tool_call 装饰器？装饰一个普通函数只会生成同步版本，
    而 server.py 用 ainvoke / astream 异步调用 agent，会直接报 NotImplementedError。
    继承 AgentMiddleware 同时实现同步和异步两个方法，命令行和 Web 服务就都能用。
    """

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            raise  # 中断（interrupt）等 LangGraph 控制流信号也是异常，必须放行
        except Exception as e:
            return self._error_message(request, e)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            return self._error_message(request, e)

    @staticmethod
    def _error_message(request, error: Exception) -> ToolMessage:
        return ToolMessage(
            content=f"工具执行出错：{error}。请向用户说明情况，不要重复调用。",
            tool_call_id=request.tool_call["id"],
        )


def build_agent(
    model: BaseChatModel,
    retriever: BaseRetriever,
    checkpointer=None,
    notes_dir: Path = NOTES_DIR,
):
    """组装 agent。依赖全部通过参数传入，测试时可以换成假模型和假检索器。"""
    tools = [
        make_search_tool(retriever),
        calculator,
        get_current_time,
        make_save_note_tool(notes_dir),
    ]
    middleware = [
        # 消息超过 40 条时，把较早的消息压缩成一段摘要，只保留最近 20 条原文
        SummarizationMiddleware(model=model, trigger=("messages", 40), keep=("messages", 20)),
        # save_note 会写磁盘，执行前暂停等用户决定；允许批准、修改参数、拒绝
        HumanInTheLoopMiddleware(
            interrupt_on={"save_note": {"allowed_decisions": ["approve", "edit", "reject"]}},
            description_prefix="StudyMate 想保存一个笔记文件",
        ),
        # 单次提问最多调用模型 8 次，超过就直接结束，防止死循环烧钱
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolErrorHandler(),
    ]
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer or InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# 命令行交互
# ---------------------------------------------------------------------------

def cli_review(request: dict) -> list[dict]:
    """在终端里让用户审批 agent 想执行的操作，返回与 action_requests 一一对应的决定列表。"""
    decisions = []
    for action in request["action_requests"]:
        args = action["args"]
        print(f"\n\n⚠️  StudyMate 想调用 {action['name']}")
        print(f"   文件名：{args.get('filename')}")
        preview = args.get("content", "")
        print(f"   内容预览：\n{preview[:300]}{'……' if len(preview) > 300 else ''}")
        choice = input("\n   [y] 批准  [e] 改文件名后保存  [n] 拒绝 > ").strip().lower()

        if choice == "e":
            new_name = input("   新文件名：").strip() or args.get("filename")
            decisions.append({
                "type": "edit",
                "edited_action": {"name": action["name"], "args": {**args, "filename": new_name}},
            })
        elif choice == "n":
            reason = input("   拒绝原因（可留空）：").strip() or "用户拒绝保存"
            decisions.append({"type": "reject", "message": reason})
        else:
            decisions.append({"type": "approve"})
    return decisions


def _messages_in(update) -> list:
    """取出节点更新里的消息。工具并行执行时，更新可能是多个 dict 组成的列表。"""
    updates = update if isinstance(update, list) else [update]
    return [m for u in updates if isinstance(u, dict) for m in u.get("messages", [])]


STREAM_MODES = ["messages", "updates"]


def to_events(mode: str, data) -> list[dict]:
    """把 LangGraph 的原始流数据翻译成前端关心的事件。命令行（run_turn）和 Web 服务（server.py）共用。

    同时订阅两种流：messages 拿逐 token 输出，updates 拿节点级事件（工具调用、中断）。
    返回的事件类型：token / tool_call / tool_result / interrupt
    """
    if mode == "messages":
        chunk, metadata = data
        # 只要主模型节点的输出：摘要中间件内部也会调用模型，它的输出不该显示给用户。
        # 支持流式的模型发来的是 AIMessageChunk（AIMessage 的子类），不支持的发来完整 AIMessage
        if metadata.get("langgraph_node") == "model" and isinstance(chunk, AIMessage) and chunk.text:
            return [{"type": "token", "text": chunk.text}]
        return []

    events = []
    for msg in _messages_in(data.get("model")):
        for call in getattr(msg, "tool_calls", None) or []:
            events.append({"type": "tool_call", "name": call["name"], "args": call["args"]})
    for msg in _messages_in(data.get("tools")):
        sources = sorted({d.metadata["source"] for d in (msg.artifact or [])}) if msg.name == "search_knowledge_base" else []
        events.append({"type": "tool_result", "name": msg.name, "chars": len(msg.content), "sources": sources})
    if "__interrupt__" in data:
        events.append({"type": "interrupt", "request": data["__interrupt__"][0].value})
    return events


def run_turn(agent, user_input: str, config: dict, review: Callable[[dict], list[dict]] = cli_review):
    """处理一轮用户输入：流式打印回答，遇到人工审批就暂停询问，然后继续。"""
    payload = {"messages": [{"role": "user", "content": user_input}]}

    while True:
        interrupt_request = None
        for mode, data in agent.stream(payload, config, stream_mode=STREAM_MODES):
            for event in to_events(mode, data):
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "tool_call":
                    args = json.dumps(event["args"], ensure_ascii=False)
                    print(f"\n  🔧 调用 {event['name']}({args[:80]}{'…' if len(args) > 80 else ''})")
                elif event["type"] == "tool_result":
                    hint = f"，命中 {', '.join(event['sources'])}" if event["sources"] else ""
                    print(f"  ↳ {event['name']} 返回 {event['chars']} 字{hint}")
                elif event["type"] == "interrupt":
                    interrupt_request = event["request"]

        print()
        if interrupt_request is None:
            return
        # 拿到用户的决定后，用 Command(resume=...) 让 agent 从暂停处继续
        payload = Command(resume={"decisions": review(interrupt_request)})


HELP = """
StudyMate —— LangChain 学习助手
可以试试：
  · LangGraph 的 checkpointer 有什么用？
  · RAG 有哪些优化手段？面试时怎么回答？
  · 我每天学 2.5 小时，学完 6 个模块每个需要 4 小时，要几天？今天开始的话哪天学完？
  · 帮我制定一份 5 天的 LangGraph 学习计划，并保存成笔记
命令：/new 开启新会话   /quit 退出
"""


def main():
    from rag_advanced import build_default_retriever  # 混合检索 + 可选重排序（第 7 课）

    print("正在同步知识库索引……")
    agent = build_agent(get_chat_model(), build_default_retriever())
    thread_id = uuid.uuid4().hex[:8]
    print(HELP)

    while True:
        try:
            user_input = input(f"\n[会话 {thread_id}] 你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/new":
            # 换一个 thread_id 就是新会话：checkpointer 按 thread_id 隔离保存状态
            thread_id = uuid.uuid4().hex[:8]
            print(f"已开启新会话 {thread_id}")
            continue

        print("StudyMate：", end="", flush=True)
        run_turn(agent, user_input, {"configurable": {"thread_id": thread_id}})

    print("\n再见！保存的笔记在 notes/ 目录。")


if __name__ == "__main__":
    main()
