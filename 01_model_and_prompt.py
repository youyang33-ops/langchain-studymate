"""第 1 课：模型、消息、提示词模板、LCEL 管道、流式输出

运行：python 01_model_and_prompt.py

学完你应该理解：
  1. 模型的输入是消息列表，输出是 AIMessage 对象
  2. 模型本身没有记忆，"多轮对话"就是把历史消息一起发过去
  3. prompt | model | parser 这种管道写法（LCEL）是怎么回事

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # langchain 仓库 libs/core/langchain_core/messages/
from langchain_core.output_parsers import StrOutputParser  # langchain 仓库 libs/core/langchain_core/output_parsers/string.py
from langchain_core.prompts import ChatPromptTemplate  # langchain 仓库 libs/core/langchain_core/prompts/chat.py

from common import get_chat_model


# 【LangChain 用法】model.invoke、AIMessage 的 content 和 usage_metadata
def demo_invoke(model):
    print("\n=== 1. invoke：最基础的调用 ===")
    response = model.invoke("用一句话解释什么是 LangChain")
    # 返回的不是字符串，而是 AIMessage 对象，里面还有 token 用量等元数据
    print(f"类型：{type(response).__name__}")
    print(f"内容：{response.content}")
    print(f"用量：{response.usage_metadata}")


# 【LangChain 用法】SystemMessage、HumanMessage、model.invoke；用来对比的"不带历史"这一步是我加的
def demo_messages(model):
    print("\n=== 2. 消息列表：多轮对话的本质 ===")
    messages = [
        SystemMessage("你是一个说话简洁的 Python 老师，每次回答不超过 50 字"),
        HumanMessage("列表和元组有什么区别？"),
    ]
    first = model.invoke(messages)
    print(f"第一轮：{first.content}")

    # 模型是无状态的：要让它"记得"上一轮，必须把上一轮的问答也放进列表
    messages += [first, HumanMessage("我刚才问了什么问题？")]
    print(f"第二轮：{model.invoke(messages).content}")

    # 反例：不带历史，模型就不知道
    print(f"不带历史：{model.invoke('我刚才问了什么问题？').content}")


# 【LangChain 用法】ChatPromptTemplate、| 管道、StrOutputParser、batch
def demo_prompt_template(model):
    print("\n=== 3. 提示词模板 + LCEL 管道 ===")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{role}，回答控制在{max_words}字以内"),
        ("human", "{question}"),
    ])

    # 先单独看模板渲染出了什么：模板的产物就是消息列表
    rendered = prompt.invoke({"role": "面试官", "max_words": 60, "question": "什么是 RAG？"})
    for msg in rendered.to_messages():
        print(f"  [{msg.type}] {msg.content}")

    # 用 | 把组件串起来：prompt 输出消息 → model 输出 AIMessage → parser 取出字符串
    # 每个组件都是 Runnable，都有 invoke / batch / stream 方法，串起来的整体也是 Runnable
    chain = prompt | model | StrOutputParser()
    print(f"\n回答：{chain.invoke({'role': '面试官', 'max_words': 60, 'question': '什么是 RAG？'})}")

    # batch：并发执行多个输入，比 for 循环逐个调用快很多
    questions = ["什么是 Agent？", "什么是 Embedding？", "什么是向量数据库？"]
    answers = chain.batch([{"role": "老师", "max_words": 30, "question": q} for q in questions])
    for q, a in zip(questions, answers):
        print(f"  {q} → {a}")


# 【LangChain 用法】model.stream、chunk.text
def demo_stream(model):
    print("\n=== 4. stream：流式输出（打字机效果）===")
    for chunk in model.stream("写一首关于程序员找工作的四行短诗"):
        # chunk 是 AIMessageChunk，多个 chunk 相加就是完整消息
        print(chunk.text, end="", flush=True)
    print()


# 【LangChain 用法】ChatPromptTemplate 里放 AIMessage 做 few-shot；分类任务和示例是我写的
def demo_few_shot(model):
    print("\n=== 5. Few-shot：用示例教模型输出格式 ===")
    # 直接在消息列表里伪造几轮"示范问答"，是最朴素也最有效的 few-shot 方法
    prompt = ChatPromptTemplate.from_messages([
        ("system", "把用户输入的技能分类为：编程语言 / 框架 / 数据库 / 其他。只输出分类结果。"),
        ("human", "Python"), AIMessage("编程语言"),
        ("human", "LangGraph"), AIMessage("框架"),
        ("human", "{skill}"),
    ])
    chain = prompt | model | StrOutputParser()
    for skill in ["Milvus", "FastAPI", "Go", "Docker"]:
        print(f"  {skill} → {chain.invoke({'skill': skill})}")


# 【我写的】把上面的例子依次跑一遍
if __name__ == "__main__":
    model = get_chat_model()
    demo_invoke(model)
    demo_messages(model)
    demo_prompt_template(model)
    demo_stream(model)
    demo_few_shot(model)
