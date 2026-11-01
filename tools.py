"""StudyMate 用到的工具。

工具 = 一个普通函数 + @tool 装饰器。LangChain 会读取：
  - 函数名            → 工具名
  - docstring         → 工具描述（模型靠它判断什么时候该调用，务必写清楚）
  - 参数类型注解      → 参数的 JSON Schema
"""

import ast
import operator
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain.tools import tool

# ---------------------------------------------------------------------------
# 计算器：用 AST 白名单求值，绝不能直接 eval 模型给的字符串（可能被注入恶意代码）
# ---------------------------------------------------------------------------

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


@tool
def get_current_time() -> str:
    """获取当前北京时间（日期、星期、时刻）。涉及"今天""几天后"等日期问题时调用。"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %A %H:%M")


# ---------------------------------------------------------------------------
# 需要外部依赖的工具用"工厂函数"创建，依赖通过闭包注入，方便测试时替换
# ---------------------------------------------------------------------------

def make_search_tool(retriever: BaseRetriever) -> BaseTool:
    """把检索器包装成工具（Agentic RAG：由模型决定何时检索、检索什么）。

    接收的是 BaseRetriever 而不是具体的向量库：简单向量检索（store.as_retriever()）
    和第 7 课的混合检索 + 重排序（HybridRetriever）都能直接传进来，工具代码不用改。
    """

    # content_and_artifact：返回 (给模型看的文本, 给程序用的原始对象)。
    # 模型只看到文本；原始 Document 列表保存在 ToolMessage.artifact 里，可用于展示引用。
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


def make_save_note_tool(notes_dir: Path) -> BaseTool:
    """创建保存笔记的工具。它会写磁盘，属于"有副作用"的操作，app.py 中会对它加人工审批。"""

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
