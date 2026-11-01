"""第 3 课：工具调用与 Agent —— 先手写，再用框架

运行：python 03_tools_and_agent.py

这一课最重要。先不用框架手写一个 agent 循环，看懂原理；
再用 create_agent 实现同样的功能，体会框架帮你省掉了什么。
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from common import get_chat_model
from tools import calculator, get_current_time


@tool
def get_salary_range(city: str, position: str) -> str:
    """查询某城市某岗位的应届生月薪范围（单位：元）。
    city: 城市名，如 北京、上海、杭州
    position: 岗位名，如 大模型应用开发、Java 后端"""
    # 演示用的假数据，真实项目里这里会调用数据库或外部 API
    table = {"北京": (18000, 30000), "上海": (17000, 28000), "杭州": (16000, 26000), "深圳": (17000, 28000)}
    low, high = table.get(city, (12000, 20000))
    if "大模型" in position or "agent" in position.lower():
        low, high = int(low * 1.2), int(high * 1.3)
    return f"{city}{position}应届月薪约 {low}-{high} 元"


TOOLS = [get_salary_range, calculator, get_current_time]
QUESTION = "北京和杭州的大模型应用开发应届生，月薪上限差多少？按 15 薪算年薪上限差多少？"


def manual_agent(model, question: str, max_steps: int = 6) -> str:
    """手写 agent 循环 —— create_agent 内部做的就是这件事。"""
    print("\n=== 1. 手写 Agent 循环 ===")
    tools_by_name = {t.name: t for t in TOOLS}

    # bind_tools：把工具的 schema 附加到每次请求上，模型由此知道有哪些工具可用
    model_with_tools = model.bind_tools(TOOLS)
    messages = [HumanMessage(question)]

    for step in range(1, max_steps + 1):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)

        # 没有 tool_calls 说明模型认为可以直接回答了，循环结束
        if not ai_msg.tool_calls:
            return ai_msg.content

        # 模型可能一次请求多个工具（并行调用），逐个执行
        for call in ai_msg.tool_calls:
            print(f"  [第 {step} 轮] 模型请求：{call['name']}({call['args']})")
            # 把整个 tool_call 传给 tool.invoke，会直接返回带正确 tool_call_id 的 ToolMessage
            tool_msg = tools_by_name[call["name"]].invoke(call)
            print(f"           工具返回：{tool_msg.content}")
            messages.append(tool_msg)

    return "超过最大轮数，任务未完成"


def framework_agent(model, question: str) -> str:
    """用 create_agent 实现同样的功能，只需要几行。"""
    print("\n=== 2. 使用 create_agent ===")
    agent = create_agent(
        model,
        tools=TOOLS,
        system_prompt="你是求职顾问。涉及数字计算必须使用 calculator 工具。",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # result["messages"] 是完整的消息轨迹：用户 → AI(请求工具) → 工具结果 → ... → AI(最终回答)
    for msg in result["messages"]:
        msg.pretty_print()
    return result["messages"][-1].content


def inspect_tool_schema():
    """看看 @tool 装饰器到底生成了什么发给模型。"""
    print("\n=== 3. 工具 schema（模型实际看到的内容）===")
    print(f"name: {get_salary_range.name}")
    print(f"description: {get_salary_range.description}")
    print(f"args: {get_salary_range.args}")


if __name__ == "__main__":
    model = get_chat_model(temperature=0)
    inspect_tool_schema()
    print(f"\n最终回答：{manual_agent(model, QUESTION)}")
    framework_agent(model, QUESTION)
