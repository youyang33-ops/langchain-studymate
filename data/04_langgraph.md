# LangGraph 核心概念

## 为什么需要 LangGraph

简单的"模型 + 工具循环"用 create_agent 就够了。但真实业务常常需要：固定的多步骤流程、条件分支、循环修改、多个 agent 协作、中途暂停等人审批、断点续跑。LangGraph 用"图"来描述这些流程，让控制流清晰可见。

## 三个基本元素

- **State（状态）**：一个 TypedDict，是整张图共享的数据。每个节点读取 state，返回需要更新的字段。
- **Node（节点）**：普通 Python 函数，输入 state，输出一个 dict（部分状态更新）。节点里可以调用模型、工具或任何代码。
- **Edge（边）**：决定执行顺序。普通边固定跳转；条件边（add_conditional_edges）根据 state 动态决定下一个节点，可以实现分支和循环。

图从 START 开始，到 END 结束，定义完需要 `compile()` 后才能运行。

## Reducer：状态如何合并

默认情况下，节点返回的字段会直接覆盖旧值。如果希望追加（例如消息列表），可以给字段加 reducer：`messages: Annotated[list, add_messages]`。add_messages 会按消息 id 去重并追加，这也是 agent 能累积对话历史的原因。

## Checkpointer：持久化与记忆

编译时传入 checkpointer（如 InMemorySaver、SqliteSaver、PostgresSaver），每执行完一步就会保存一次 state 快照。调用时通过 `config={"configurable": {"thread_id": "xxx"}}` 指定会话：

- 同一个 thread_id 下的多次调用共享历史，实现多轮对话记忆。
- 不同 thread_id 互相隔离，相当于不同用户或不同会话。
- 可以用 `get_state` 查看当前状态，用 `get_state_history` 回溯历史，实现"时间旅行"调试。

InMemorySaver 只存在内存里，程序重启就没了；生产环境用数据库版本。

## Interrupt：人工介入

在节点里调用 `interrupt(payload)` 会暂停整张图，把 payload 返回给调用方。人工处理后用 `graph.invoke(Command(resume=答复), config)` 从暂停点继续。这依赖 checkpointer，因为暂停时的状态必须保存下来。HumanInTheLoopMiddleware 就是基于这个机制实现的。

## 流式输出的几种模式

- `stream_mode="updates"`：每个节点执行完输出该节点的状态更新，适合展示进度。
- `stream_mode="values"`：每步输出完整 state。
- `stream_mode="messages"`：逐 token 输出模型生成内容，适合聊天界面的打字机效果。
- 可以传列表同时订阅多种模式。

## 多 Agent 常见架构

- **Supervisor（主管）模式**：一个主 agent 负责分派任务给多个专业子 agent，再汇总结果。
- **Handoff（交接）模式**：agent 之间直接移交控制权，例如客服 agent 转给退款 agent。
- **子图**：把一个编译好的图作为另一个图的节点，便于模块化复用。

## LangChain 与 LangGraph 的区别（面试常问）

LangChain 提供组件和高层封装（模型接口、工具、create_agent、中间件），追求快速上手；LangGraph 是底层编排运行时，提供状态管理、持久化、中断、流式等能力，追求可控性。create_agent 构建在 LangGraph 之上，当 create_agent 的固定循环不能满足需求时，就直接用 LangGraph 自己画图。
