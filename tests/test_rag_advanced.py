"""第 7 课进阶 RAG 组件的测试。

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

import httpx  # 第三方库 httpx，不属于 LangChain
import pytest  # 第三方库 pytest，不属于 LangChain
from langchain_core.documents import Document  # langchain 仓库 libs/core/langchain_core/documents/base.py
from langchain_core.embeddings import DeterministicFakeEmbedding  # langchain 仓库 libs/core/langchain_core/embeddings/fake.py

import rag_advanced
from rag_advanced import (
    BM25Index,
    DashScopeReranker,
    HybridRetriever,
    LLMReranker,
    build_chroma,
    chunk_id,
    evaluate_retrieval,
    is_hit,
    load_eval_cases,
    load_knowledge_base,
    multi_query_retrieve,
    reciprocal_rank_fusion,
    rewrite_query,
    tokenize,
)
from tests.fakes import fake_vector_store, scripted, tool_call


# 【我写的】小工具：创建 Document
def doc(text: str, source: str = "a.md") -> Document:
    return Document(page_content=text, metadata={"source": source})


# ---------------------------------------------------------------------------
# 加载与评估集
# ---------------------------------------------------------------------------

# 【我写的】手写一个最小的 PDF，测试用
def make_pdf(path, text: str):
    """手写一个最小的单页 PDF（只含英文，因为标准 Helvetica 字体没有中文字形）。"""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(out)


# 【我写的】测试：PDF 会进入知识库
def test_knowledge_base_includes_pdf_pages(tmp_path):
    (tmp_path / "notes.md").write_text("# 标题\n\n正文内容", encoding="utf-8")
    make_pdf(tmp_path / "paper.pdf", "Hybrid search combines BM25 and dense vectors")

    chunks = load_knowledge_base(tmp_path)
    pdf_chunks = [c for c in chunks if c.metadata["source"] == "paper.pdf"]
    assert len(pdf_chunks) == 1
    assert pdf_chunks[0].metadata["page"] == 1
    assert "BM25" in pdf_chunks[0].page_content


# 【我写的】测试：评估集每条用例只对应一个答案片段
def test_every_eval_case_has_exactly_one_answer_chunk():
    chunks = load_knowledge_base()
    for case in load_eval_cases():
        assert sum(is_hit(c, case) for c in chunks) == 1, case["question"]


# 【我写的】测试：命中率和 MRR 的计算
def test_evaluate_retrieval_metrics():
    cases = [
        {"question": "q1", "source": "a.md", "keyword": "答案"},
        {"question": "q2", "source": "a.md", "keyword": "答案"},
        {"question": "q3", "source": "a.md", "keyword": "答案"},
    ]
    results = {
        "q1": [doc("答案")],                # 排第 1 → 1
        "q2": [doc("无关"), doc("答案")],    # 排第 2 → 1/2
        "q3": [doc("无关")],                # 未命中 → 0
    }
    r = evaluate_retrieval(lambda q: results[q], cases)
    assert r["hit_rate"] == pytest.approx(2 / 3)
    assert r["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert r["misses"] == ["q3"]


# ---------------------------------------------------------------------------
# BM25 与 RRF
# ---------------------------------------------------------------------------

# 【我写的】测试：分词结果
def test_tokenize_mixes_words_and_chinese_bigrams():
    assert tokenize("RAG 召回率 HNSW") == ["rag", "召回", "回率", "hnsw"]


# 【我写的】测试：BM25 排序和 IDF
def test_bm25_prefers_rare_matching_terms():
    docs = [doc("向量 检索 介绍"), doc("向量 数据库 HNSW 索引"), doc("向量 向量 向量")]
    bm25 = BM25Index(docs)
    assert bm25.idf["hnsw"] > bm25.idf["向量"]  # 越少见的词区分度越高
    top_doc, _ = bm25.search("HNSW 索引", k=1)[0]
    assert "HNSW" in top_doc.page_content
    assert bm25.search("完全不相关的词", k=3) == []


# 【我写的】测试：RRF 融合排序
def test_rrf_rewards_documents_found_by_both_retrievers():
    a, b, c = doc("A"), doc("B"), doc("C")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0].page_content == "B"  # 两路都召回的 B 排第一，即使它在每一路都只排第 2
    assert len(fused) == 3               # 重复文档合并


# ---------------------------------------------------------------------------
# 混合检索、重排序、查询改写
# ---------------------------------------------------------------------------

# 【我写的】测试用的假重排序器
class ReverseReranker:
    # 【我写的】把结果倒序
    def rerank(self, query, docs, top_n):
        return list(reversed(docs))[:top_n]


# 【我写的】测试：混合检索和重排序的流程
def test_hybrid_retriever_fuses_then_reranks():
    chunks = load_knowledge_base()
    store = fake_vector_store()
    plain = HybridRetriever(vector_store=store, bm25=BM25Index(chunks), top_k=3)
    reranked = HybridRetriever(vector_store=store, bm25=BM25Index(chunks), reranker=ReverseReranker(), top_k=3)

    query = "proxy_buffering 参数"
    assert len(plain.invoke(query)) == 3
    # BM25 能精确命中 proxy_buffering，所以融合结果里一定有它
    assert any("proxy_buffering" in d.page_content for d in plain.invoke(query))
    assert reranked.invoke(query) != plain.invoke(query)


# 【我写的】测试：rerank 接口的请求和响应格式
def test_dashscope_reranker_request_and_response(monkeypatch):
    sent = {}

    # 【我写的】代替真实网络请求
    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        body = {"output": {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.12}]}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(rag_advanced.httpx, "post", fake_post)
    reranker = DashScopeReranker(model="qwen3-rerank", api_key="test-key", url="https://example.test/rerank")
    result = reranker.rerank("问题", [doc("第一段"), doc("第二段")], top_n=2)

    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["json"] == {
        "model": "qwen3-rerank",
        "input": {"query": "问题", "documents": ["第一段", "第二段"]},
        "parameters": {"top_n": 2},
    }
    assert [d.page_content for d in result] == ["第二段", "第一段"]
    assert result[0].metadata["rerank_score"] == 0.91


# 【我写的】测试：按模型打分排序
def test_llm_reranker_sorts_by_model_scores():
    model = scripted(tool_call("_DocScores", {"scores": [
        {"index": 0, "score": 2}, {"index": 1, "score": 9}, {"index": 2, "score": 5},
    ]}))
    result = LLMReranker(model).rerank("q", [doc("低"), doc("高"), doc("中")], top_n=2)
    assert [d.page_content for d in result] == ["高", "中"]


# 【我写的】测试：查询改写和多路合并
def test_rewrite_and_multi_query():
    model = scripted(tool_call("RewrittenQuery", {
        "standalone": "LangGraph 生产环境用哪种 checkpointer",
        "variants": ["PostgresSaver", "checkpointer 持久化"],
    }))
    rewritten = rewrite_query(model, "那生产环境用哪种？", ["checkpointer 有什么用"])
    assert "checkpointer" in rewritten.standalone

    retriever = fake_vector_store().as_retriever(search_kwargs={"k": 3})
    docs = multi_query_retrieve(retriever, [rewritten.standalone, *rewritten.variants], top_k=4)
    assert len(docs) == 4
    assert len({chunk_id(d) for d in docs}) == 4  # 合并后没有重复


# ---------------------------------------------------------------------------
# 持久化与增量索引
# ---------------------------------------------------------------------------

# 【我写的】测试：增量索引的新增、修改、删除
def test_chroma_incremental_indexing(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)
    chunks = [doc("第一段"), doc("第二段"), doc("第三段")]

    store = build_chroma(chunks, embeddings, persist_dir=tmp_path)
    assert len(store.get(include=[])["ids"]) == 3

    # 修改一段、删除一段、新增一段
    updated = [doc("第一段"), doc("第二段（已修改）"), doc("第四段")]
    store = build_chroma(updated, embeddings, persist_dir=tmp_path)
    assert set(store.get(include=[])["ids"]) == {chunk_id(c) for c in updated}


# 【我写的】测试：内容不变时不重新计算向量
def test_chroma_skips_embedding_for_unchanged_chunks(tmp_path):
    # 【我写的】统计 Embedding 调用次数
    class CountingEmbedding(DeterministicFakeEmbedding):
        calls: int = 0

        # 【我写的】计数后调用父类方法
        def embed_documents(self, texts):
            self.calls += len(texts)
            return super().embed_documents(texts)

    embeddings = CountingEmbedding(size=32)
    chunks = [doc("甲"), doc("乙")]
    build_chroma(chunks, embeddings, persist_dir=tmp_path)
    build_chroma(chunks, embeddings, persist_dir=tmp_path)  # 第二次：内容没变
    assert embeddings.calls == 2
