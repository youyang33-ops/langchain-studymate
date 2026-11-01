# LangChain 生态总览

## 各个包分别做什么

LangChain 不是一个单独的库，而是一组分工明确的包：

- **langchain-core**：最底层的抽象。定义了 ChatModel、Message、PromptTemplate、Tool、Runnable、Document、VectorStore 等接口。所有其他包都依赖它。
- **langchain**：1.x 版本后变得很精简，核心是 `create_agent` 和 middleware（中间件），用于快速搭建 agent。
- **langgraph**：底层编排框架，用"状态 + 节点 + 边"的图来描述工作流，支持持久化、中断恢复、人工介入。`create_agent` 本身就是用 LangGraph 实现的。
- **langchain-openai / langchain-anthropic / langchain-deepseek 等**：各模型厂商的集成包（partners）。国产模型大多兼容 OpenAI 协议，用 langchain-openai 改 base_url 即可接入。
- **langchain-text-splitters**：文本切分工具，做 RAG 时用。
- **langchain-community**：社区维护的大量集成（文档加载器、向量库等），质量参差不齐。
- **langchain-classic**：1.0 之前的旧 API（如 LLMChain、旧式 Retriever 链）被挪到了这里，新项目不建议使用。
- **LangSmith**：不是开源库，而是官方的观测平台，用于链路追踪（tracing）、评估（evaluation）和提示词管理。

## 1.x 和旧版本的主要区别

- 旧版的 `initialize_agent`、`AgentExecutor`、`LLMChain` 已经废弃，统一用 `create_agent`。
- 引入 middleware 机制，可以在模型调用前后、工具调用前后插入自定义逻辑。
- 消息新增了 `content_blocks` 标准化内容格式，统一不同厂商的多模态和推理内容。
- Python 最低版本要求 3.10。

## 源码阅读建议

GitHub 仓库 langchain-ai/langchain 是一个 monorepo：

- `libs/core/langchain_core/runnables/base.py`：Runnable 协议和 LCEL 管道符 `|` 的实现。
- `libs/core/langchain_core/tools/`：`@tool` 装饰器和 BaseTool。
- `libs/langchain_v1/langchain/agents/factory.py`：`create_agent` 的实现，可以看到它如何把模型节点、工具节点和中间件组装成一张 LangGraph 图。
- `libs/langchain_v1/langchain/agents/middleware/`：所有内置中间件，例如人工审批、对话摘要、调用次数限制。
- `libs/partners/openai/`：ChatOpenAI 的实现，了解厂商集成怎么写。
