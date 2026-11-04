"""第 5 课：记忆、中间件、人工审批

运行：python 05_memory_and_middleware.py

  1. 记忆：checkpointer + thread_id 实现多轮对话和会话隔离
  2. 中间件：在 agent 循环中插入自定义逻辑（日志、动态提示词、工具错误处理）
  3. 人工审批：敏感工具执行前暂停，等人确认后继续

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

from dataclasses import dataclass

from langchain.agents import create_agent  # langchain 仓库 libs/langchain_v1/langchain/agents/factory.py
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,  # langchain 仓库 libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py
    ModelRequest,  # langchain 仓库 libs/langchain_v1/langchain/agents/middleware/types.py
    before_model,  # langchain 仓库 libs/langchain_v1/langchain/agents/middleware/types.py
    dynamic_prompt,  # langchain 仓库 libs/langchain_v1/langchain/agents/middleware/types.py
    wrap_tool_call,  # langchain 仓库 libs/langchain_v1/langchain/agents/middleware/types.py
)
from langchain.tools import tool  # langchain 仓库 libs/core/langchain_core/tools/convert.py
from langchain_core.messages import ToolMessage  # langchain 仓库 libs/core/langchain_core/messages/tool.py
from langgraph.checkpoint.memory import InMemorySaver  # 来自 langchain-ai/langgraph 仓库，不在 langchain 仓库
from langgraph.types import Command  # 来自 langchain-ai/langgraph 仓库，不在 langchain 仓库

from common import ROOT, get_chat_model
from tools import make_save_note_tool


# 【我写的】小工具：拼出 agent 要求的消息字典
def user(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# 1. 记忆
# ---------------------------------------------------------------------------

# 【LangChain + LangGraph 接口 + 我写的】LangChain：create_agent；LangGraph：InMemorySaver、thread_id、get_state；我写的：三组对比实验
def demo_memory(model):
    print("\n=== 1. 记忆：checkpointer + thread_id ===")

    no_memory = create_agent(model, tools=[])
    no_memory.invoke(user("我叫小林，正在找 Agent 开发岗位"))
    print(f"无记忆：{no_memory.invoke(user('我叫什么？在找什么工作？'))['messages'][-1].content}")

    # checkpointer 会在每一步之后保存 state，下次用同一个 thread_id 调用时自动恢复
    agent = create_agent(model, tools=[], checkpointer=InMemorySaver())
    thread_a = {"configurable": {"thread_id": "user-a"}}
    thread_b = {"configurable": {"thread_id": "user-b"}}

    agent.invoke(user("我叫小林，正在找 Agent 开发岗位"), thread_a)
    print(f"同一会话：{agent.invoke(user('我叫什么？在找什么工作？'), thread_a)['messages'][-1].content}")
    print(f"另一会话：{agent.invoke(user('我叫什么？'), thread_b)['messages'][-1].content}")

    # 可以直接查看某个会话保存的完整状态
    state = agent.get_state(thread_a)
    print(f"会话 user-a 已保存 {len(state.values['messages'])} 条消息")


# ---------------------------------------------------------------------------
# 2. 自定义中间件
# ---------------------------------------------------------------------------

# 【我写的】字段是我设计的，传给 LangChain 的 context_schema
@dataclass
class UserContext:
    """运行时上下文：每次调用时传入，不会进入对话历史。适合放用户身份、权限、偏好等。"""
    level: str  # "beginner" 或 "expert"


# 【LangChain 接口 + 我写的】LangChain：@dynamic_prompt、request.runtime.context；我写的：按水平切换提示词
@dynamic_prompt
def prompt_by_level(request: ModelRequest) -> str:
    """每次调用模型前，根据用户水平动态生成系统提示词。"""
    if request.runtime.context.level == "beginner":
        return "你在给编程初学者讲解。多用生活中的比喻，避免术语，不超过 100 字。"
    return "你在和资深工程师交流。直接讲实现原理和关键 API，不超过 100 字。"


# 【LangChain 接口 + 我写的】LangChain：@before_model；我写的：打印日志
@before_model
def log_before_model(state, runtime) -> None:
    """每次调用模型前打印日志。返回 None 表示不修改 state；返回 dict 则会更新 state。"""
    print(f"  [middleware] 即将调用模型，当前共 {len(state['messages'])} 条消息")


# 【LangChain 接口 + 我写的】LangChain：@wrap_tool_call、ToolMessage；我写的：捕获异常并返回错误信息
@wrap_tool_call
def catch_tool_errors(request, handler):
    """包裹每一次工具调用：工具抛异常时转成一条错误消息返回给模型，而不是让整个程序崩溃。

    注意：装饰普通函数只生成同步版本，只能配合 invoke / stream 使用。
    异步调用（ainvoke / astream）需要装饰 async 函数，或像 app.py 的 ToolErrorHandler 那样继承 AgentMiddleware。
    """
    try:
        return handler(request)
    except Exception as e:
        print(f"  [middleware] 工具 {request.tool_call['name']} 出错：{e!r}")
        return ToolMessage(
            content=f"工具调用失败：{e}。请不要重试，直接告诉用户该服务暂时不可用。",
            tool_call_id=request.tool_call["id"],
        )


# 【LangChain 接口 + 我写的】LangChain：@tool；我写的：故意报错，用来演示错误处理
@tool
def query_job_board(keyword: str) -> str:
    """在招聘网站搜索岗位。keyword: 岗位关键词"""
    raise ConnectionError("招聘网站接口超时")  # 故意出错，演示错误处理中间件


# 【LangChain 用法】create_agent 的 middleware、context_schema 参数和 invoke 的 context 参数
def demo_middleware(model):
    print("\n=== 2. 自定义中间件 ===")
    agent = create_agent(
        model,
        tools=[query_job_board],
        middleware=[prompt_by_level, log_before_model, catch_tool_errors],
        context_schema=UserContext,
    )

    question = user("什么是向量数据库？")
    for level in ["beginner", "expert"]:
        result = agent.invoke(question, context=UserContext(level=level))
        print(f"[{level}] {result['messages'][-1].content}\n")

    result = agent.invoke(user("帮我搜一下 Agent 开发岗位"), context=UserContext(level="beginner"))
    print(f"工具出错后的回答：{result['messages'][-1].content}")


# ---------------------------------------------------------------------------
# 3. 人工审批（Human-in-the-loop）
# ---------------------------------------------------------------------------

# 【LangChain + LangGraph 接口 + 我写的】LangChain：HumanInTheLoopMiddleware 和 decisions 格式（参考 human_in_the_loop.py 里的示例）；LangGraph：InMemorySaver、Command；我写的：终端审批交互
def demo_human_in_the_loop(model):
    print("\n=== 3. 人工审批 ===")
    agent = create_agent(
        model,
        tools=[make_save_note_tool(ROOT / "notes")],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_note": True})],
        checkpointer=InMemorySaver(),  # 必须有：暂停时的状态要保存下来，恢复时才能继续
    )
    config = {"configurable": {"thread_id": "hitl-demo"}}

    result = agent.invoke(user("帮我写 3 条 RAG 面试要点，保存为笔记 rag面试要点"), config)

    # 模型想调用 save_note 时，图会暂停，结果里出现 __interrupt__
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"模型想调用 {action['name']}，参数：")
            print(f"  filename = {action['args']['filename']}")
            print(f"  content  = {action['args']['content'][:200]}")
        answer = input("是否批准？(y/n)：").strip().lower()
        decision = {"type": "approve"} if answer == "y" else {"type": "reject", "message": "用户不想保存"}
        # 用 Command(resume=...) 从暂停点继续执行，decisions 与 action_requests 一一对应
        result = agent.invoke(Command(resume={"decisions": [decision] * len(request["action_requests"])}), config)

    print(f"最终回答：{result['messages'][-1].content}")


# 【我写的】把上面的例子依次跑一遍
if __name__ == "__main__":
    model = get_chat_model()
    demo_memory(model)
    demo_middleware(model)
    demo_human_in_the_loop(model)
