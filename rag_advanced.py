"""进阶 RAG 组件（第 7 课的核心代码，app.py 和 server.py 也在用）

  1. PDF 加载
  2. 持久化向量库（Chroma）+ 增量索引：只为新增或改动的 chunk 调用 Embedding
  3. BM25 关键词检索（手写实现，面试常考公式）
  4. 混合检索：向量 + BM25，用 RRF 融合排名
  5. 重排序（Rerank）：通义 qwen3-rerank 接口，或用大模型打分兜底
  6. 查询改写：消除多轮对话中的指代，并生成多个检索语句
  7. 检索评估：命中率（Hit Rate）和 MRR
"""

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from common import DATA_DIR, ROOT, get_embeddings, load_documents, split_documents

CHROMA_DIR = ROOT / ".chroma"
EVAL_FILE = ROOT / "eval" / "retrieval_cases.jsonl"


# ---------------------------------------------------------------------------
# 1. 加载：Markdown + PDF
# ---------------------------------------------------------------------------

def load_pdf(path: Path) -> list[Document]:
    """每一页一个 Document，metadata 记录页码，回答时可以引用"第几页"。

    注意：pypdf 只能提取文字层。扫描件需要 OCR；表格、公式多的 PDF
    可以考虑 MinerU、Unstructured 等专门的文档解析工具。
    """
    from pypdf import PdfReader

    return [
        Document(page_content=text, metadata={"source": path.name, "page": i})
        for i, page in enumerate(PdfReader(path).pages, 1)
        if (text := page.extract_text().strip())
    ]


def load_knowledge_base(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = load_documents(data_dir)
    for pdf in sorted(data_dir.glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return split_documents(docs)


def chunk_id(doc: Document) -> str:
    """用"来源 + 内容"的哈希作为 chunk 的唯一 id：内容不变 id 就不变，这是增量索引的基础。"""
    return hashlib.sha1(f"{doc.metadata.get('source')}\n{doc.page_content}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. 持久化向量库 + 增量索引
# ---------------------------------------------------------------------------

def build_chroma(
    chunks: list[Document],
    embeddings: Embeddings | None = None,
    persist_dir: Path = CHROMA_DIR,
    collection: str = "studymate_kb",
) -> Chroma:
    """打开（或创建）本地 Chroma 向量库，并与当前 chunks 同步。

    第 4 课的 InMemoryVectorStore 每次启动都要重新调用 Embedding 接口，既慢又花钱。
    这里把向量存到磁盘，下次启动只处理有变化的部分：
      - 新增或修改过的 chunk → 计算向量并写入
      - 已删除的 chunk       → 从库中删除
      - 没变的 chunk         → 直接复用
    """
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir),
        # Chroma 底层用 HNSW 图索引做近似最近邻搜索，这里指定用余弦距离
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    current = {chunk_id(c): c for c in chunks}  # 用 dict 去重：内容完全相同的 chunk 只存一份
    existing = set(store.get(include=[])["ids"])

    stale = existing - current.keys()
    new_ids = [i for i in current if i not in existing]
    if stale:
        store.delete(ids=list(stale))
    if new_ids:
        store.add_documents([current[i] for i in new_ids], ids=new_ids)
    print(f"[索引] 新增 {len(new_ids)}，删除 {len(stale)}，复用 {len(existing) - len(stale)}")
    return store


# ---------------------------------------------------------------------------
# 3. BM25 关键词检索
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """英文和数字按单词切分，中文切成相邻两字一组（bigram）。

    例如 "RAG 召回率" → ["rag", "召回", "回率"]。bigram 不需要词典，效果也不错；
    生产环境中文通常用 jieba 分词，或 Elasticsearch 的 IK 分词器。
    """
    tokens = []
    for piece in _TOKEN_RE.findall(text.lower()):
        if piece.isascii() or len(piece) == 1:
            tokens.append(piece)
        else:
            tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))
    return tokens


class BM25Index:
    """BM25 打分公式（面试常考）：

        score(q, D) = Σ IDF(t) · tf(t,D)·(k1+1) / (tf(t,D) + k1·(1 - b + b·|D|/avgdl))
        IDF(t)      = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

      tf(t,D) 词 t 在文档 D 中出现的次数    n(t) 包含词 t 的文档数    N 文档总数
      k1      控制词频饱和度：同一个词出现 10 次，得分并不是出现 1 次的 10 倍
      b       控制文档长度归一化：长文档天然包含更多词，需要惩罚
    """

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.term_freqs = [Counter(tokenize(d.page_content)) for d in docs]
        self.doc_lens = [sum(tf.values()) for tf in self.term_freqs]
        self.avgdl = sum(self.doc_lens) / max(len(docs), 1)
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(docs)
        self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in doc_freq.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, norm = self.term_freqs[i], 1 - self.b + self.b * self.doc_lens[i] / self.avgdl
        return sum(
            self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * norm)
            for t in query_tokens if t in tf
        )

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        tokens = tokenize(query)
        scored = [(self.docs[i], self.score(tokens, i)) for i in range(len(self.docs))]
        scored = [pair for pair in scored if pair[1] > 0]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# 4. 混合检索：RRF 融合
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(result_lists: Iterable[list[Document]], k: int = 60) -> list[Document]:
    """RRF：score(d) = Σ 1 / (k + rank_i(d))

    向量检索的分数（余弦相似度）和 BM25 分数量纲不同，不能直接相加。
    RRF 只看排名不看分数，简单稳健，是混合检索最常用的融合方法。
    k=60 是论文中的经验值，作用是削弱"排第 1 和排第 2"之间的差距。
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            key = chunk_id(doc)
            scores[key] += 1 / (k + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]: ...


class HybridRetriever(BaseRetriever):
    """自定义检索器：向量召回 + BM25 召回 → RRF 融合 →（可选）重排序。

    继承 BaseRetriever 并实现 _get_relevant_documents，就能像官方检索器一样
    使用 invoke / batch / 管道符 |，也能直接传给 make_search_tool。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: VectorStore
    bm25: BM25Index
    reranker: Reranker | None = None
    fetch_k: int = 10  # 每一路召回的数量（粗排）
    top_k: int = 4     # 最终返回的数量（精排）

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        dense = self.vector_store.similarity_search(query, k=self.fetch_k)
        sparse = [doc for doc, _ in self.bm25.search(query, k=self.fetch_k)]
        fused = reciprocal_rank_fusion([dense, sparse])
        if self.reranker:
            return self.reranker.rerank(query, fused, top_n=self.top_k)
        return fused[:self.top_k]


# ---------------------------------------------------------------------------
# 5. 重排序
# ---------------------------------------------------------------------------

class DashScopeReranker:
    """调用阿里云百炼的 qwen3-rerank 接口。

    Embedding 是"双塔"（Bi-Encoder）：问题和文档分别编码成向量再比较，速度快，可以预先算好，但精度一般。
    Rerank 模型是"交叉编码"（Cross-Encoder）：把问题和文档拼在一起输入模型直接判断相关性，
    精度高，但每对都要算一遍，所以只用来对召回的几十条结果做精排。
    """

    DEFAULT_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(self, model: str | None = None, api_key: str | None = None, url: str | None = None):
        self.model = model or os.getenv("RERANK_MODEL", "qwen3-rerank")
        self.api_key = api_key or os.getenv("RERANK_API_KEY") or os.getenv("LLM_API_KEY")
        self.url = url or os.getenv("RERANK_URL") or self.DEFAULT_URL

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": {"query": query, "documents": [d.page_content for d in docs]},
                "parameters": {"top_n": top_n},
            },
            timeout=30,
        )
        resp.raise_for_status()
        # 返回结果已按相关性从高到低排序，index 指向输入 documents 中的位置
        return [
            _with_score(docs[r["index"]], r["relevance_score"])
            for r in resp.json()["output"]["results"]
        ]


class _DocScore(BaseModel):
    index: int = Field(description="资料编号")
    score: int = Field(ge=0, le=10, description="与问题的相关性，0=无关，10=直接回答了问题")


class _DocScores(BaseModel):
    scores: list[_DocScore]


class LLMReranker:
    """没有专用 rerank 接口时，让大模型给每个片段打分。比专用模型慢且贵，适合演示或兜底。"""

    def __init__(self, model: BaseChatModel):
        self.scorer = model.with_structured_output(_DocScores, method="function_calling")

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        numbered = "\n\n".join(f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs))
        result = self.scorer.invoke(
            f"判断每条资料与问题的相关性并逐条打分（0-10）。\n问题：{query}\n\n资料：\n{numbered}"
        )
        scores = {s.index: s.score for s in result.scores if 0 <= s.index < len(docs)}
        ranked = sorted(range(len(docs)), key=lambda i: scores.get(i, 0), reverse=True)
        return [_with_score(docs[i], scores.get(i, 0) / 10) for i in ranked[:top_n]]


def _with_score(doc: Document, score: float) -> Document:
    return Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": round(score, 4)})


def get_reranker() -> Reranker | None:
    """配置了 RERANK_MODEL 就启用专用重排序模型，否则不重排。"""
    return DashScopeReranker() if os.getenv("RERANK_MODEL") else None


def build_default_retriever(embeddings: Embeddings | None = None, reranker: Reranker | None = None) -> HybridRetriever:
    """app.py 和 server.py 使用的生产级检索器。"""
    chunks = load_knowledge_base()
    return HybridRetriever(
        vector_store=build_chroma(chunks, embeddings),
        bm25=BM25Index(chunks),
        reranker=reranker or get_reranker(),
    )


# ---------------------------------------------------------------------------
# 6. 查询改写
# ---------------------------------------------------------------------------

class RewrittenQuery(BaseModel):
    standalone: str = Field(description="结合对话历史改写成的独立、完整的问题，消除'它''这个''那'等指代")
    variants: list[str] = Field(description="2-3 个换了表述或侧重点的检索语句，用于多路召回")


def rewrite_query(model: BaseChatModel, question: str, history: list[str] = ()) -> RewrittenQuery:
    """多轮对话中用户常说"那它呢"，直接拿这句话去检索什么都找不到，要先改写。"""
    rewriter = model.with_structured_output(RewrittenQuery, method="function_calling")
    context = "\n".join(f"- {h}" for h in history) or "（无）"
    return rewriter.invoke(f"对话历史：\n{context}\n\n用户最新问题：{question}\n\n请改写成适合知识库检索的语句。")


def multi_query_retrieve(retriever: BaseRetriever, queries: list[str], top_k: int = 4) -> list[Document]:
    """多个检索语句分别检索，再用 RRF 合并：提高召回率，降低单次表述不准带来的漏检。"""
    return reciprocal_rank_fusion(retriever.batch(queries))[:top_k]


# ---------------------------------------------------------------------------
# 7. 检索评估
# ---------------------------------------------------------------------------

def load_eval_cases(path: Path = EVAL_FILE) -> list[dict]:
    """每条用例：问题 + 正确答案所在的文件 + 该 chunk 中必须出现的关键词。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_hit(doc: Document, case: dict) -> bool:
    return doc.metadata.get("source") == case["source"] and case["keyword"].lower() in doc.page_content.lower()


def evaluate_retrieval(search: Callable[[str], list[Document]], cases: list[dict]) -> dict:
    """Hit Rate：正确 chunk 出现在结果中的比例。
    MRR（平均倒数排名）：正确 chunk 排第 1 得 1 分、排第 2 得 1/2 分……取平均，衡量排得靠不靠前。"""
    hits, reciprocal_ranks, misses = 0, [], []
    for case in cases:
        rank = next((i for i, d in enumerate(search(case["question"]), 1) if is_hit(d, case)), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
            misses.append(case["question"])
    return {
        "hit_rate": hits / len(cases),
        "mrr": sum(reciprocal_ranks) / len(cases),
        "misses": misses,
    }


if __name__ == "__main__":
    # 单独运行本文件：只构建 / 同步一次索引
    build_default_retriever()
    print("索引已同步到", CHROMA_DIR)
