"""第 4 课：RAG 检索增强生成

运行：python 04_rag.py

知识库是 data/ 目录下的 LangChain 学习笔记，你可以往里面加自己的 .md 文件。
流程：加载 → 切分 → 向量化 → 存储（见 common.py）→ 检索 → 生成（本文件）
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from common import build_vector_store, get_chat_model, load_documents, split_documents


def demo_split():
    print("\n=== 1. 切分：看看文档被切成了什么样 ===")
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"{len(docs)} 个文件 → {len(chunks)} 个 chunk")
    for c in chunks[:3]:
        print(f"\n  metadata: {c.metadata}")
        print(f"  内容（前 80 字）: {c.page_content[:80]!r}")


def demo_retrieve(store):
    print("\n=== 2. 检索：向量相似度搜索 ===")
    query = "agent 一直重复调用工具停不下来怎么办"
    # 分数越高越相似（InMemoryVectorStore 用余弦相似度）
    for doc, score in store.similarity_search_with_score(query, k=3):
        print(f"  [{score:.3f}] {doc.metadata['source']} / {doc.metadata.get('h3') or doc.metadata.get('h2')}")


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i}] 来源：{d.metadata['source']}\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


def build_rag_chain(store, model):
    """固定链路 RAG：每个问题都先检索再生成。"""
    retriever = store.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是 LangChain 学习助手。只根据下面提供的资料回答问题，"
         "并在相应句子后用 [编号] 标注引用。资料中没有的信息，直接回答"
         "\"资料中没有相关内容\"，不要编造。"),
        ("human", "资料：\n{context}\n\n问题：{question}"),
    ])
    # 字典会被自动转成 RunnableParallel：两个分支并行执行，结果合并成 dict 交给 prompt
    #   context  分支：问题 → retriever 检索出 Document 列表 → format_docs 拼成字符串
    #   question 分支：RunnablePassthrough 原样传递问题
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def demo_rag(store, model):
    print("\n=== 3. RAG 问答 ===")
    chain = build_rag_chain(store, model)
    for q in [
        "LangGraph 的 checkpointer 有什么用？thread_id 是做什么的？",
        "RAG 的 chunk 大小应该怎么选？",
        "LangChain 的创始人是谁？",  # 知识库里没有，测试模型会不会编造
    ]:
        print(f"\n问：{q}\n答：{chain.invoke(q)}")


if __name__ == "__main__":
    demo_split()
    print("\n正在调用 Embedding 接口构建向量库……")
    store = build_vector_store()
    demo_retrieve(store)
    demo_rag(store, get_chat_model(temperature=0))
