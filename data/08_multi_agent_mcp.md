# 多 Agent 与 MCP

## 什么时候需要多 Agent

单个 Agent 在工具不多、任务单一时效果最好。出现以下情况时考虑拆分：

- 工具太多（通常超过 10 到 20 个），模型选错工具的概率明显上升；
- 不同子任务需要完全不同的系统提示词和专业知识；
- 单个 Agent 的上下文被大量中间结果撑满；
- 子任务可以并行执行，以降低总耗时。

不要为了"多 Agent"而多 Agent：每多一层 Agent 就多一次模型调用，延迟、成本和出错点都会增加。

## 常见架构

### Supervisor（主管）模式

一个主管 Agent 负责理解任务、分派给专业子 Agent、汇总结果。在 LangChain 1.x 中推荐的实现方式是把子 Agent 包装成工具（subagents as tools）：主管通过工具调用来"雇佣"子 Agent。

优点是上下文隔离：子 Agent 内部的多轮工具调用不会进入主管的上下文，主管只看到最终结果。缺点是所有信息都要经过主管中转，链路较长。

### Handoff（交接）模式

Agent 之间直接移交控制权，例如前台客服 Agent 判断是退款问题后，把对话交给退款 Agent，之后用户直接和退款 Agent 对话。适合客服等需要"转接"的场景。在 LangGraph 中可以通过工具返回 Command(goto=...) 实现跳转。

### 工作流 + 并行（Map-Reduce）

流程固定时，不需要让模型来决定步骤，直接用 LangGraph 画图更可控。例如"提取技能列表 → 每个技能并行调研 → 汇总报告"：用 Send API 为每个技能动态创建一个并行分支，结果通过 reducer（例如 operator.add）合并到同一个列表字段，最后由汇总节点统一处理。

### 如何选择

| | Supervisor（模型决策） | 工作流（代码决策） |
|---|---|---|
| 灵活性 | 高，能处理意料之外的请求 | 低，只能走预设流程 |
| 可预测性 | 低，每次路径可能不同 | 高，便于测试和排查 |
| 成本与延迟 | 较高，主管每一步都要调用模型 | 较低，可以并行 |
| 适用场景 | 开放式任务、对话助手 | 流程明确的业务，如报告生成、审批 |

实际项目常常混合使用：外层是确定的工作流，某些节点内部是自主的 Agent。

## 多 Agent 的常见问题

- **信息丢失**：子 Agent 的结果被压缩汇报，细节丢失。可以要求子 Agent 按固定结构返回。
- **重复劳动或死循环**：主管反复派发同样的任务。需要限制调用次数，并在提示词中明确完成条件。
- **成本失控**：每一层都在调用模型。监控每个 Agent 的 token 用量。
- **难以调试**：用 LangSmith 等链路追踪工具查看完整调用树。

## MCP（Model Context Protocol）

MCP 是 Anthropic 在 2024 年底提出的开放协议，用来统一"AI 应用如何连接外部工具和数据"。可以类比为 AI 领域的 USB-C 接口：工具方按 MCP 实现一次服务端，就能被所有支持 MCP 的客户端（Claude、Cursor、各种 Agent 框架）使用，不用为每个框架各写一遍。

### 核心概念

- **Host**：用户直接使用的 AI 应用，例如 Claude Desktop、IDE。
- **Client**：Host 内部负责与某个 MCP Server 通信的连接器。
- **Server**：对外提供能力的服务，暴露三类东西：Tools（可调用的函数）、Resources（可读取的数据，如文件、数据库记录）、Prompts（预设的提示词模板）。
- **传输方式**：本地进程用 stdio，远程服务用 Streamable HTTP。

### Function Calling 和 MCP 的关系

Function Calling 是模型层面的能力：模型输出"我要调用哪个函数、参数是什么"。MCP 是应用层面的协议：规定工具如何被发现、描述和调用。两者是互补关系：MCP Server 提供的工具，最终还是通过 Function Calling 让模型来决定是否调用。

### 在 LangChain 中使用 MCP

使用 langchain-mcp-adapters 包的 MultiServerMCPClient 连接一个或多个 MCP Server，调用 get_tools() 得到 LangChain 工具列表，直接传给 create_agent 即可。

### MCP 的安全问题

MCP Server 返回的内容会进入模型上下文，恶意的 Server 或被污染的数据可能包含提示词注入。应只接入可信的 Server，对有副作用的工具加人工审批，并遵循最小权限原则。

## A2A 协议

A2A（Agent2Agent）是 Google 在 2025 年提出的开放协议，关注的是 Agent 和 Agent 之间如何互相发现（通过 Agent Card 描述自己的能力）、委派任务和交换结果。简单区分：MCP 解决 Agent 连接工具和数据，A2A 解决 Agent 与 Agent 之间的协作。
