"""StudyMate 用到的工具。

工具 = 一个普通函数 + @tool 装饰器。LangChain 会读取：
  - 函数名            → 工具名
  - docstring         → 工具描述（模型靠它判断什么时候该调用，务必写清楚）
  - 参数类型注解      → 参数的 JSON Schema

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

import ast
import operator
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.documents import Document  # langchain 仓库 libs/core/langchain_core/documents/base.py
from langchain_core.retrievers import BaseRetriever  # langchain 仓库 libs/core/langchain_core/retrievers.py
from langchain_core.tools import BaseTool  # langchain 仓库 libs/core/langchain_core/tools/base.py
from langchain.tools import tool  # langchain 仓库 libs/core/langchain_core/tools/convert.py

# ---------------------------------------------------------------------------
# 计算器：用 AST 白名单求值，绝不能直接 eval 模型给的字符串（可能被注入恶意代码）
# ---------------------------------------------------------------------------

# 【我写的】计算器允许的运算符白名单
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# 【我写的】用 Python 标准库 ast 做安全求值
def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("只支持数字、括号和 + - * / % ** 运算")


# 【LangChain 接口 + 我写的】LangChain：@tool；我写的：计算逻辑
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持 + - * / % ** 和括号，例如 '(3+5)*2' 或 '120/7'。
    任何需要计算的场景都应调用此工具，不要心算。"""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        # 返回错误文字而不是抛异常：模型能读到错误原因并自行修正参数
        return f"计算失败：{e}"
    return str(round(result, 6))


# 【LangChain 接口 + 我写的】LangChain：@tool；我写的：取北京时间
@tool
def get_current_time() -> str:
    """获取当前北京时间（日期、星期、时刻）。涉及"今天""几天后"等日期问题时调用。"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %A %H:%M")


# ---------------------------------------------------------------------------
# 需要外部依赖的工具用"工厂函数"创建，依赖通过闭包注入，方便测试时替换
# ---------------------------------------------------------------------------

# 【LangChain 接口 + 我写的】LangChain：BaseRetriever；我写的：把检索器包装成工具的工厂函数
def make_search_tool(retriever: BaseRetriever) -> BaseTool:
    """把检索器包装成工具（Agentic RAG：由模型决定何时检索、检索什么）。

    接收的是 BaseRetriever 而不是具体的向量库：简单向量检索（store.as_retriever()）
    和第 7 课的混合检索 + 重排序（HybridRetriever）都能直接传进来，工具代码不用改。
    """

    # content_and_artifact：返回 (给模型看的文本, 给程序用的原始对象)。
    # 模型只看到文本；原始 Document 列表保存在 ToolMessage.artifact 里，可用于展示引用。
    # 【LangChain 接口 + 我写的】LangChain：@tool(response_format="content_and_artifact")、retriever.invoke；我写的：工具描述和结果格式
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str) -> tuple[str, list[Document]]:
        """在学习笔记知识库中检索资料。知识库涵盖 LangChain 生态、Agent 与工具调用、RAG 与向量检索、
        LangGraph、多 Agent 与 MCP、大模型基础、工程化部署、Agent 岗位面试题。回答这些主题的问题前必须先调用。
        query: 检索语句，应是完整、具体的问题或关键词，不要用"它""这个"等指代词。"""
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中没有找到相关内容。", []
        text = "\n\n".join(f"[来源: {d.metadata['source']}]\n{d.page_content}" for d in docs)
        return text, docs

    return search_knowledge_base


# 【我写的】工厂函数，把保存目录注入工具
def make_save_note_tool(notes_dir: Path) -> BaseTool:
    """创建保存笔记的工具。它会写磁盘，属于"有副作用"的操作，app.py 中会对它加人工审批。"""

    # 【LangChain 接口 + 我写的】LangChain：@tool；我写的：写文件和文件名清洗
    @tool
    def save_note(filename: str, content: str) -> str:
        """把内容保存为 Markdown 笔记文件，例如学习计划、知识点总结、面试题整理。
        filename: 文件名，不需要扩展名，例如 "langgraph学习计划"
        content: Markdown 格式的完整笔记内容"""
        # 文件名来自模型输出，必须清洗，防止 "../../" 之类的路径穿越
        safe_name = re.sub(r"[^\w-]", "_", filename).strip("_")[:50] or "note"
        notes_dir.mkdir(parents=True, exist_ok=True)
        path = notes_dir / f"{safe_name}.md"
        path.write_text(content, encoding="utf-8")
        return f"笔记已保存：{path.name}（{len(content)} 字）"

    return save_note
