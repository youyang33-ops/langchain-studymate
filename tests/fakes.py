"""测试共用的假对象：剧本式假模型、假检索器。"""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore

from rag_advanced import load_knowledge_base


class ScriptedModel(GenericFakeChatModel):
    """按顺序返回预设消息的假模型。"""

    # GenericFakeChatModel 流式输出时会丢掉 tool_calls，所以关闭流式，整条消息一次返回
    disable_streaming: bool = True

    def bind_tools(self, tools, **kwargs):
        return self  # 假模型不需要真的知道工具，按剧本返回 tool_calls 即可

    def with_structured_output(self, schema, **kwargs):
        kwargs.pop("method", None)
        return super().with_structured_output(schema, **kwargs)


def scripted(*messages) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def fake_vector_store() -> InMemoryVectorStore:
    """用确定性假向量建库：检索结果没有语义意义，但能验证流程是否跑通。"""
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=256))
    store.add_documents(load_knowledge_base())
    return store
