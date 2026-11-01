"""公共模块：模型初始化 + 知识库构建。所有课程脚本和 app.py 都从这里获取模型。

配置全部来自 .env 文件（复制 .env.example 改名为 .env 再填写）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼（通义千问）


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}：请把 .env.example 复制为 .env 并填写")
    return value


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def get_chat_model(temperature: float = 0.3) -> BaseChatModel:
    """返回聊天模型。

    init_chat_model 是 1.x 推荐的统一入口：换模型只需改参数，业务代码不用动。
    国产模型（通义、DeepSeek、智谱、Kimi……）大多兼容 OpenAI 协议，
    所以 model_provider 统一用 "openai"，再通过 base_url 指向各家的接口地址。
    """
    return init_chat_model(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        model_provider="openai",
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=_require_env("LLM_API_KEY"),
        temperature=temperature,
    )


def get_embeddings() -> Embeddings:
    """返回 Embedding 模型（把文本转成向量，RAG 检索用）。"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("EMBEDDING_API_KEY") or _require_env("LLM_API_KEY"),
        # 坑 1：OpenAIEmbeddings 默认先用 tiktoken 把文本转成 token id 再发送，
        #       非 OpenAI 的兼容接口不认 token id，必须关掉，改为直接发送原文。
        check_embedding_ctx_length=False,
        # 坑 2：通义 text-embedding-v3 每次请求最多 10 条文本，默认的 1000 会报错。
        chunk_size=10,
    )


# ---------------------------------------------------------------------------
# 知识库（RAG 的离线索引阶段：加载 → 切分 → 向量化 → 存储）
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """加载 data/ 下所有 Markdown 文件。

    这里为了少装依赖手写了加载逻辑。处理 PDF、Word、网页时，
    可以用 langchain-community 里现成的 PyPDFLoader、WebBaseLoader 等。
    """
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(data_dir.glob("*.md"))
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """两级切分：先按 Markdown 标题切成章节，章节过长时再按字符切。"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,  # 保留标题文字，检索时标题本身也是重要信号
    )
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # 中文没有空格分词，要把中文标点加进分隔符，否则可能从句子中间切断
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata.update(doc.metadata)  # 把来源文件名带到每个 chunk 上
            chunks.extend(chunk_splitter.split_documents([section]))
    return chunks


def build_vector_store(embeddings: Embeddings | None = None) -> InMemoryVectorStore:
    """构建内存向量库。生产环境可替换为 Chroma、Milvus、pgvector，接口一致。"""
    store = InMemoryVectorStore(embeddings or get_embeddings())
    store.add_documents(split_documents(load_documents()))
    return store
