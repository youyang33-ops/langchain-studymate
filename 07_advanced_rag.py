"""第 7 课：进阶 RAG —— 混合检索、重排序、查询改写、用数据评估效果

运行：python 07_advanced_rag.py

组件实现都在 rag_advanced.py，本文件负责演示和对比。
这一课最重要的是第 4 部分：用测试集量化每种方案的效果。
面试时说"我用了混合检索"不稀奇，说"混合检索把命中率从 X% 提升到 Y%"才有说服力。
"""

from langchain_core.documents import Document

from common import get_chat_model
from rag_advanced import (
    BM25Index,
    DashScopeReranker,
    HybridRetriever,
    LLMReranker,
    build_chroma,
    evaluate_retrieval,
    get_reranker,
    load_eval_cases,
    load_knowledge_base,
    multi_query_retrieve,
    rewrite_query,
    tokenize,
)


def show(docs: list[Document], n: int = 3):
    for d in docs[:n]:
        score = f" rerank={d.metadata['rerank_score']}" if "rerank_score" in d.metadata else ""
        section = d.metadata.get("h3") or d.metadata.get("h2") or d.metadata.get("h1")
        print(f"    · {d.metadata['source']} / {section}{score}")


def demo_vector_vs_bm25(store, bm25):
    print("\n=== 1. 向量检索 vs BM25：各有擅长 ===")
    print(f"BM25 分词示例：{tokenize('Nginx 流式输出 proxy_buffering')}")
    cases = [
        ("精确术语", "proxy_buffering 参数"),
        ("口语化同义表达", "文档更新了不想全部重新算向量"),
    ]
    for kind, query in cases:
        print(f"\n[{kind}] {query}")
        print("  向量检索：")
        show(store.similarity_search(query, k=3))
        print("  BM25：")
        show([d for d, _ in bm25.search(query, k=3)])


def demo_hybrid_and_rerank(store, bm25, reranker):
    print("\n=== 2. 混合检索 + 重排序 ===")
    query = "怎么让检索结果排得更准"
    hybrid = HybridRetriever(vector_store=store, bm25=bm25)
    print(f"[{query}]\n  混合检索（RRF 融合）：")
    show(hybrid.invoke(query))
    if reranker:
        print(f"  混合检索 + {type(reranker).__name__}：")
        show(HybridRetriever(vector_store=store, bm25=bm25, reranker=reranker).invoke(query))


def demo_query_rewrite(model, retriever):
    print("\n=== 3. 查询改写 ===")
    history = ["用户：LangGraph 的 checkpointer 有什么用？", "助手：用于保存每一步的状态，实现记忆和中断恢复……"]
    question = "那生产环境应该用哪种？"
    rewritten = rewrite_query(model, question, history)
    print(f"原问题：{question}")
    print(f"改写后：{rewritten.standalone}")
    print(f"多查询：{rewritten.variants}")
    print("  直接用原问题检索：")
    show(retriever.invoke(question))
    print("  用改写后的多个查询检索并融合：")
    show(multi_query_retrieve(retriever, [rewritten.standalone, *rewritten.variants]))


def demo_evaluation(store, bm25, reranker):
    print("\n=== 4. 用测试集评估（eval/retrieval_cases.jsonl）===")
    cases = load_eval_cases()
    candidates = {
        "向量检索": lambda q: store.similarity_search(q, k=4),
        "BM25": lambda q: [d for d, _ in bm25.search(q, k=4)],
        "混合检索 (RRF)": HybridRetriever(vector_store=store, bm25=bm25).invoke,
    }
    if reranker:
        candidates["混合 + Rerank"] = HybridRetriever(vector_store=store, bm25=bm25, reranker=reranker).invoke

    print(f"共 {len(cases)} 条用例，取前 4 条结果\n")
    print(f"  {'方案':<16}{'命中率':>8}{'MRR':>8}")
    for name, search in candidates.items():
        r = evaluate_retrieval(search, cases)
        print(f"  {name:<16}{r['hit_rate']:>8.1%}{r['mrr']:>8.3f}")
        for q in r["misses"]:
            print(f"      未命中：{q}")


if __name__ == "__main__":
    chunks = load_knowledge_base()
    store = build_chroma(chunks)  # 第一次运行会调用 Embedding，之后复用磁盘上的 .chroma/
    bm25 = BM25Index(chunks)
    model = get_chat_model(temperature=0)

    # 配置了 RERANK_MODEL 就用专用重排序模型，否则用大模型打分来演示原理
    reranker = get_reranker() or LLMReranker(model)
    if isinstance(reranker, DashScopeReranker):
        print(f"使用重排序模型：{reranker.model}")
    else:
        print("未配置 RERANK_MODEL，用大模型打分代替重排序模型（较慢）")

    demo_vector_vs_bm25(store, bm25)
    demo_hybrid_and_rerank(store, bm25, reranker)
    demo_query_rewrite(model, HybridRetriever(vector_store=store, bm25=bm25))
    demo_evaluation(store, bm25, reranker)
