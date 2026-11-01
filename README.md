# StudyMate：边做项目边学 LangChain 1.x

一个循序渐进的 LangChain 学习项目：9 节课 + 1 个综合项目。最终做出一个**带混合检索 RAG、工具调用、人工审批、多轮记忆，并部署为流式 Web 服务的学习助手 Agent**。

基于 `langchain 1.4` / `langgraph 1.2` / `fastapi 0.162` 编写并测试，全部使用 1.x 新 API（`create_agent`、middleware），不含任何已废弃写法。

## 快速开始

**需要 Python 3.11 或更高版本**（原因见文末"踩坑记录"第 8 条）。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # 然后编辑 .env，填入 API Key
pytest                      # 36 个离线测试，不需要 Key，确认环境正常
python 01_model_and_prompt.py
```

推荐用**阿里云百炼（通义千问）**：一个 Key 同时提供聊天模型、Embedding 和重排序模型，新用户有免费额度。也可以用 DeepSeek，配置方法见 `.env.example`。

## 项目结构

```
├── common.py                     模型初始化 + 知识库加载与切分
├── tools.py                      工具定义：计算器、时间、知识库检索、保存笔记
├── rag_advanced.py               进阶 RAG 组件：Chroma、BM25、混合检索、重排序、查询改写、评估
│
├── 01_model_and_prompt.py        第 1 课：模型、消息、提示词模板、LCEL、流式
├── 02_structured_output.py       第 2 课：结构化输出（学习计划生成、JD 信息抽取）
├── 03_tools_and_agent.py         第 3 课：手写 Agent 循环 vs create_agent  ⭐
├── 04_rag.py                     第 4 课：RAG 基础
├── 05_memory_and_middleware.py   第 5 课：记忆、中间件、人工审批
├── 06_langgraph_workflow.py      第 6 课：LangGraph 自定义工作流（写作-审稿循环）
├── 07_advanced_rag.py            第 7 课：进阶 RAG + 检索效果评估  ⭐
├── 08_multi_agent.py             第 8 课：多 Agent（Supervisor vs Map-Reduce）
├── server.py + static/           第 9 课：FastAPI 部署（SSE 流式 + 网页聊天界面）
│
├── app.py                        综合项目：命令行版 StudyMate
├── data/                         知识库：9 篇学习笔记，同时也是面试复习资料
├── eval/retrieval_cases.jsonl    检索评估测试集（24 条）
└── tests/                        离线测试：用假模型验证所有流程
```

## 学习路线

### 第一阶段：LangChain 基础（第 1–6 课）

每一课都**先读代码注释，再运行，最后做练习**。

| 课 | 核心概念 | 运行时重点观察 | 练习 |
|---|---|---|---|
| 1 | `invoke` / 消息列表 / `ChatPromptTemplate` / `\|` 管道 / `stream` / `batch` | 不带历史消息时模型为什么"失忆" | 写一个链：输入英文 JD，输出中文翻译和 3 个关键词 |
| 2 | `with_structured_output` + Pydantic | `include_raw` 里原始的 `tool_calls`：结构化输出其实是伪装成工具调用 | 定义 `Resume` 模型，从自我介绍里抽取简历信息 |
| 3 | `@tool` / `bind_tools` / `ToolMessage` / `create_agent` | 手写循环和 `pretty_print()` 的消息轨迹是一样的 | 加一个 `get_company_info` 工具 |
| 4 | 切分 / Embedding / 向量检索 / RAG 链 | 知识库没有的问题，模型会不会编造 | 把 `chunk_size` 改成 200 和 1000 对比效果 |
| 5 | `checkpointer` + `thread_id` / 中间件 / 人工审批 | `[middleware]` 日志的打印时机 | 写一个 `@after_model` 中间件统计回答字数 |
| 6 | `StateGraph` / 条件边 / 循环 | 每一稿的分数；把 Mermaid 图贴到 mermaid.live | 加一个 `polish` 节点 |

第 3 课的 `manual_agent()` 用不到 20 行手写了 Agent 的核心循环，`create_agent` 做的就是这件事。面试问"不用框架怎么实现 Agent"，答案就在这里。

### 第二阶段：面试高频进阶（第 7–9 课）

#### 第 7 课：进阶 RAG

```bash
python 07_advanced_rag.py
```

| 技术 | 实现位置 | 面试要点 |
|---|---|---|
| PDF 加载 | `load_pdf` | 往 `data/` 放 PDF 会自动纳入知识库；扫描件需要 OCR |
| Chroma 持久化 + 增量索引 | `build_chroma` | 内容哈希作为 id，只为变化的部分调用 Embedding |
| BM25 | `BM25Index`（手写，约 30 行） | 能写出公式，讲清 k1、b、IDF 的作用 |
| 混合检索 + RRF | `HybridRetriever`、`reciprocal_rank_fusion` | 为什么分数不能直接相加 |
| 重排序 | `DashScopeReranker`（qwen3-rerank）、`LLMReranker` | 双塔与交叉编码器的区别 |
| 查询改写 | `rewrite_query`、`multi_query_retrieve` | 指代消解、多查询召回 |
| 检索评估 | `evaluate_retrieval` + `eval/retrieval_cases.jsonl` | 命中率、MRR |

**最重要的是第 4 部分的评估表**。它会对比"向量 / BM25 / 混合 / 混合 + Rerank"四种方案的命中率和 MRR。把你跑出来的真实数字写进简历，比"使用了混合检索"有说服力得多。练习：往测试集里加 10 条你自己的问题，看哪种方案最稳。

#### 第 8 课：多 Agent

```bash
python 08_multi_agent.py
```

同一个需求（输入 JD → 生成面试准备报告），用两种架构实现并对比模型调用次数和耗时：

- **Supervisor**：主管 Agent 把三个子 Agent（JD 分析师、研究员、面试官）当作工具调用。重点理解**上下文隔离**：研究员内部的检索过程不会进入主管的上下文。
- **Map-Reduce 工作流**：用 `Send` 为每个技能创建并行分支，用 reducer（`operator.add`）合并结果。

练习：把你真实想投的 JD 贴进去，看哪种方案的报告更好。

#### 第 9 课：FastAPI 部署

```bash
uvicorn server:app --reload
```

打开 http://127.0.0.1:8000 使用聊天界面，或打开 http://127.0.0.1:8000/docs 查看自动生成的接口文档。

| 接口 | 说明 |
|---|---|
| `POST /chat` | 非流式，一次性返回 |
| `POST /chat/stream` | SSE 流式：`start` / `token` / `tool_call` / `tool_result` / `approval_required` / `error` / `done` |
| `POST /chat/resume` | 提交审批决定后继续执行（SSE） |
| `GET /threads/{id}` | 会话历史 + 待审批操作 |

用 curl 体验流式接口：

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"message": "RAG 有哪些优化手段？"}'
```

和命令行版相比的关键变化：全部异步（`ainvoke` / `astream`）；会话存进 SQLite，重启不丢；人工审批改为"推送审批事件 → 前端展示 → 调用 resume 接口"两段式。

## 综合项目：StudyMate

命令行版 `python app.py` 和网页版 `uvicorn server:app` 共用同一个 agent（`app.build_agent`）。

| 需求 | 实现方式 | 课程 |
|---|---|---|
| 基于知识库回答并标注来源 | 混合检索 + 重排序，包装为检索工具 | 4、7 |
| 计算、日期交给工具 | `calculator`（AST 安全求值）、`get_current_time` | 3 |
| 保存笔记，保存前用户确认 | `save_note` + `HumanInTheLoopMiddleware` | 3、5 |
| 多轮对话，会话持久化 | `InMemorySaver`（命令行）/ `AsyncSqliteSaver`（网页） | 5、9 |
| 长对话自动摘要、防死循环、工具出错不崩溃 | 内置中间件 + 自定义 `ToolErrorHandler` | 5 |
| 流式输出 + 展示工具调用过程 | `to_events()` 统一解析，命令行打印 / SSE 推送 | 1、9 |

## 知识库（data/）也是面试复习资料

| 文件 | 内容 |
|---|---|
| 01–04 | LangChain 生态、Agent 与工具、RAG 基础、LangGraph |
| 05_interview.md | 面试高频问答 + **StudyMate 项目怎么讲** |
| 06_llm_basics.md | Transformer、注意力、token、KV Cache、采样参数、幻觉、RAG vs 微调、LoRA |
| 07_vector_retrieval.md | Embedding、HNSW/IVF、BM25、混合检索、Rerank、查询改写、RAG 评估 |
| 08_multi_agent_mcp.md | 多 Agent 架构对比、MCP 概念与安全、A2A |
| 09_engineering.md | FastAPI、SSE vs WebSocket、异步、持久化、稳定性、成本、安全、部署 |

复习时可以直接问 StudyMate，它会从这些笔记里检索并标注来源。

## 离线测试

```bash
pytest -v
```

36 个测试覆盖：工具安全性、Agent 流程、人工审批（批准、修改、拒绝）、会话隔离、BM25 和 RRF 算法、Chroma 增量索引、重排序接口协议、多 Agent 委派与上下文隔离、并行分支合并、HTTP 接口、SSE 事件、服务重启后恢复待审批状态。全部用"按剧本回复的假模型"，不需要 API Key。

## 对照源码阅读

| 想弄懂 | 看这里（langchain-ai/langchain 仓库） |
|---|---|
| `\|` 管道符的实现 | `libs/core/langchain_core/runnables/base.py` |
| `@tool` 如何生成 schema | `libs/core/langchain_core/tools/convert.py` |
| 自定义检索器的基类 | `libs/core/langchain_core/retrievers.py` |
| `create_agent` 如何组装成图 | `libs/langchain_v1/langchain/agents/factory.py` |
| 人工审批中间件 | `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py` |
| ChatOpenAI 如何对接接口 | `libs/partners/openai/langchain_openai/chat_models/base.py` |

## 踩坑记录（都已在代码中处理）

1. **国产 Embedding 接口报错**：`OpenAIEmbeddings` 需设置 `check_embedding_ctx_length=False`；通义 v3 每批最多 10 条，需设置 `chunk_size=10`。
2. **结构化输出报错**：部分国产模型不支持默认的 `json_schema` 模式，改用 `method="function_calling"`。
3. **`InMemoryVectorStore` 需要 numpy**。
4. **中文切分**：分隔符要加入 `。！？；，`。
5. **流式输出混入了摘要内容**：按 `metadata["langgraph_node"] == "model"` 过滤。
6. **checkpointer 读回的 artifact 变成 dict**：持久化后 `Document` 对象被序列化了。
7. **`@wrap_tool_call` 装饰普通函数后不能异步调用**：`ainvoke` / `astream` 时会报 `NotImplementedError`。需要同时支持同步和异步时，继承 `AgentMiddleware` 并实现两个方法（见 `app.ToolErrorHandler`）。
8. **Python 3.10 下异步人工审批报错**：`Called get_config outside of a runnable context`。3.11 之前 contextvars 不会自动传入异步任务，LangGraph 的 `interrupt()` 拿不到配置。解决：使用 Python 3.11+。
9. **SSE 接口里抛 `HTTPException` 无效**：流一旦开始，状态码 200 已经发出。前置校验要放进 FastAPI 依赖（见 `require_pending_approval`）。
10. **`gte-rerank` 已于 2026-05-30 下线**：改用 `qwen3-rerank`。
11. **Chroma 集合名限制**：只能用 3–512 个英文字母、数字、`.`、`_`、`-`。

## 下一步扩展

- 接入 MCP 工具（`langchain-mcp-adapters`）
- 用 LangSmith 搭建端到端评估：不只评估检索，还评估回答的忠实度
- 用 LangGraph Store 实现跨会话的长期记忆（记住用户的技术栈和薄弱点）
- Docker 打包，SQLite 换 PostgreSQL，支持多实例部署
- 给 Web 服务加用户认证和限流
