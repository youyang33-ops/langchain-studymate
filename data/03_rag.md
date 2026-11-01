# RAG 检索增强生成

## 为什么需要 RAG

大模型的知识有截止日期，也不知道企业内部资料。RAG（Retrieval-Augmented Generation）的思路是：先从知识库检索出相关片段，再把片段和问题一起交给模型回答。这样能减少幻觉，并且回答可以标注出处。

## 标准流程

离线索引阶段：

1. **加载（Load）**：把 PDF、Markdown、网页等读成 Document 对象，每个 Document 有 page_content 和 metadata（如来源文件名、页码）。
2. **切分（Split）**：文档太长放不进上下文，需要切成小块（chunk）。常用 RecursiveCharacterTextSplitter，按段落、句子逐级切分。Markdown 可以先用 MarkdownHeaderTextSplitter 按标题切，保留章节信息。
3. **向量化（Embed）**：用 Embedding 模型把每个 chunk 转成向量。中文场景常用通义 text-embedding、智谱 embedding、BGE 系列等。
4. **存储（Store）**：存进向量数据库。学习阶段用 InMemoryVectorStore 或 Chroma，生产环境常用 Milvus、pgvector、Elasticsearch。

在线问答阶段：

5. **检索（Retrieve）**：把问题也转成向量，找出最相似的 top-k 个 chunk。
6. **生成（Generate）**：把检索结果拼进提示词，让模型基于资料回答。

## chunk 大小怎么选

chunk 太大，检索不精准，还会浪费 token；chunk 太小，丢失上下文，答案不完整。中文一般从 300 到 800 字开始尝试，设置 10% 左右的重叠（chunk_overlap），防止关键句被切断。

## 常见优化手段（面试高频）

- **混合检索**：向量检索（语义）+ BM25 关键词检索，对专有名词、型号、人名效果更好。
- **重排序（Rerank）**：先粗召回 20 条，再用 rerank 模型（如 bge-reranker）精排取前 5 条。
- **查询改写**：把用户口语化或多轮对话中的指代问题，改写成独立、完整的检索语句。
- **元数据过滤**：按部门、时间、文档类型先过滤再检索。
- **父子文档**：用小 chunk 检索保证精准，返回它所在的大段落保证上下文完整。

## 两种 RAG 形态

- **固定链路 RAG**：每个问题都检索一次，流程固定，延迟可控，用 LCEL 管道实现即可。
- **Agentic RAG**：把检索做成一个工具交给 agent，由模型决定要不要检索、检索几次、用什么关键词。更灵活，但更难控制成本和稳定性。

## 如何评估 RAG

分别评估检索和生成：检索看召回率、命中率（MRR）；生成看忠实度（回答是否都有资料支撑）、答案相关性。可以构建一批"问题-标准答案-相关文档"测试集，用 LangSmith 或 Ragas 做自动化评估。
