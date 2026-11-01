# Agent 与工具调用

## 什么是 Agent

Agent 是"让大模型自己决定下一步做什么"的程序。最经典的模式是 ReAct（Reasoning + Acting）：模型先思考，再决定调用某个工具，拿到工具结果后继续思考，直到能给出最终答案。

## Tool Calling（函数调用）的完整流程

1. 把工具的名称、描述、参数 JSON Schema 随请求一起发给模型。
2. 模型如果认为需要工具，会返回一个带 `tool_calls` 的 AIMessage，里面包含工具名、参数和调用 id。注意：模型只是"提出调用请求"，真正执行函数的是我们的代码。
3. 程序执行工具，把结果包装成 ToolMessage（必须带上对应的 tool_call_id）追加到消息列表。
4. 再次调用模型，模型看到工具结果后，要么继续调用工具，要么给出最终回答。
5. 循环直到模型不再请求工具，或者达到最大轮数。

工具描述（docstring）写得好不好，直接决定模型会不会在正确的时机调用正确的工具。参数名要有含义，描述要说明什么时候该用这个工具。

## create_agent

LangChain 1.x 中 `create_agent(model, tools, system_prompt=..., middleware=[...], checkpointer=...)` 帮你实现了上面的循环。它返回一个编译好的 LangGraph 图，包含 model 节点和 tools 节点，调用方式是 `agent.invoke({"messages": [...]})`。

## Middleware（中间件）

中间件是 1.x 最重要的新特性，可以在 agent 循环的各个环节插入逻辑：

- `before_model` / `after_model`：模型调用前后执行，例如裁剪消息、记录日志、检查输出。
- `wrap_model_call`：包裹模型调用，可以实现重试、模型降级、动态切换模型。
- `wrap_tool_call`：包裹工具调用，可以统一做错误处理、权限检查、缓存。
- `dynamic_prompt`：根据运行时上下文动态生成系统提示词。

常用内置中间件：

- `HumanInTheLoopMiddleware`：敏感工具执行前暂停，等人工批准、修改或拒绝。
- `SummarizationMiddleware`：对话太长时自动把早期消息压缩成摘要，防止超出上下文窗口。
- `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware`：限制调用次数，防止 agent 陷入死循环、烧钱。
- `ModelFallbackMiddleware`：主模型失败时自动切换到备用模型。
- `PIIMiddleware`：检测并脱敏手机号、邮箱等隐私信息。

## 结构化输出

让模型输出符合 Pydantic 模型的数据，常见两种实现：

- **Tool 策略**：把 schema 伪装成一个工具让模型"调用"，兼容性最好，几乎所有支持函数调用的模型都能用。
- **Provider 策略**：使用厂商原生的 JSON Schema 模式，更可靠但不是所有模型都支持。

单独使用时用 `model.with_structured_output(Schema)`；在 agent 中用 `create_agent(..., response_format=Schema)`。
