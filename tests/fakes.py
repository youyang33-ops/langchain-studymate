"""测试共用的假对象：剧本式假模型、假检索器。

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

from langchain_core.embeddings import DeterministicFakeEmbedding  # langchain 仓库 libs/core/langchain_core/embeddings/fake.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # langchain 仓库 libs/core/langchain_core/language_models/fake_chat_models.py
from langchain_core.messages import AIMessage  # langchain 仓库 libs/core/langchain_core/messages/ai.py
from langchain_core.vectorstores import InMemoryVectorStore  # langchain 仓库 libs/core/langchain_core/vectorstores/in_memory.py

from rag_advanced import load_knowledge_base


# 【LangChain 接口 + 我写的】LangChain：GenericFakeChatModel 假模型基类；我写的：让它支持工具调用测试
class ScriptedModel(GenericFakeChatModel):
    """按顺序返回预设消息的假模型。"""

    # GenericFakeChatModel 流式输出时会丢掉 tool_calls，所以关闭流式，整条消息一次返回
    disable_streaming: bool = True

    # 【LangChain 接口 + 我写的】LangChain：覆盖基类方法；我写的：直接返回自身
    def bind_tools(self, tools, **kwargs):
        return self  # 假模型不需要真的知道工具，按剧本返回 tool_calls 即可

    # 【LangChain 接口 + 我写的】LangChain：覆盖基类方法；我写的：去掉基类不接受的 method 参数
    def with_structured_output(self, schema, **kwargs):
        kwargs.pop("method", None)
        return super().with_structured_output(schema, **kwargs)


# 【我写的】小工具：创建假模型
def scripted(*messages) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


# 【LangChain 接口 + 我写的】LangChain：AIMessage 的 tool_calls 格式；我写的：小工具
def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# 【LangChain 用法】InMemoryVectorStore、DeterministicFakeEmbedding
def fake_vector_store() -> InMemoryVectorStore:
    """用确定性假向量建库：检索结果没有语义意义，但能验证流程是否跑通。"""
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=256))
    store.add_documents(load_knowledge_base())
    return store
