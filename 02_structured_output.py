"""第 2 课：结构化输出 —— 让模型返回 Python 对象而不是一段文字

运行：python 02_structured_output.py

真实业务里，模型的输出往往要交给下游程序处理（存数据库、调接口、渲染页面），
自由文本没法可靠解析。with_structured_output 让模型直接输出符合 Pydantic 定义的数据。

代码来源标注（每个函数和类前面都有一行，第三方 import 的行尾写了出处）：
  【LangChain 用法】          基本只是调用 LangChain 的接口，例子里的文字和数据是我（Claude）写的
  【LangChain 接口 + 我写的】 用了 LangChain 的接口，逻辑或提示词是我写的（标注里写了各是哪部分）
  【LangGraph ……】            含义同上，但 LangGraph 在另一个仓库 langchain-ai/langgraph，不在 langchain 仓库
  【我写的】                  主要逻辑是我写的，没有依赖 LangChain / LangGraph 的功能（最多用到 Document 这类数据类型）
"来自 LangChain"指用了它的接口和写法，这个项目没有整段复制 langchain 仓库里的代码。
"""

from pydantic import BaseModel, Field  # 第三方库 Pydantic，不属于 LangChain

from common import get_chat_model


# Field 的 description 会作为 schema 的一部分发给模型，相当于"填表说明"，要写清楚
# 【我写的】字段是我设计的，BaseModel 和 Field 来自 Pydantic
class DailyTask(BaseModel):
    day: int = Field(description="第几天，从 1 开始")
    topic: str = Field(description="当天学习主题")
    hours: float = Field(description="预计学习时长（小时）")
    deliverable: str = Field(description="当天要完成的可检验产出，例如'跑通 RAG demo'")


# 【我写的】字段是我设计的
class StudyPlan(BaseModel):
    goal: str = Field(description="学习目标的一句话总结")
    tasks: list[DailyTask]
    tips: list[str] = Field(description="3 条以内的学习建议")


# 【我写的】字段是我设计的
class JobPosting(BaseModel):
    """从招聘 JD 中抽取的结构化信息"""
    company: str = Field(description="公司名称")
    title: str = Field(description="岗位名称")
    city: str | None = Field(default=None, description="工作城市，没写则为空")
    min_years: int | None = Field(default=None, description="要求的最低工作年限，应届或未写则为空")
    must_have: list[str] = Field(description="硬性要求的技能")
    nice_to_have: list[str] = Field(description="加分项技能")


# 【我写的】虚构的招聘信息
JD_TEXT = """
【某某科技】大模型应用开发工程师（Agent 方向）— 北京
岗位职责：负责基于大模型的智能体应用研发，包括 RAG 知识库问答、多 Agent 工作流编排。
任职要求：
1. 本科及以上学历，2026 届应届生或 1 年以内经验；
2. 熟练使用 Python，熟悉 LangChain / LangGraph 至少一种框架；
3. 了解向量数据库（Milvus、Elasticsearch 等）和 Embedding 原理；
4. 有 Dify、Coze 使用经验者优先；有大模型微调经验者优先。
"""


# 【LangChain 用法】with_structured_output(method="function_calling")；打印结果的部分是我写的
def demo_study_plan(model):
    print("\n=== 1. 生成结构化学习计划 ===")
    # method="function_calling"：把 schema 当作一个工具让模型"调用"，兼容性最好。
    # 默认的 json_schema 模式需要模型原生支持，部分国产模型不支持。
    planner = model.with_structured_output(StudyPlan, method="function_calling")
    plan = planner.invoke("我有 Python 基础，想用 5 天入门 LangChain 开发，每天能学 2 小时")

    # plan 是真正的 StudyPlan 对象，可以直接用属性访问，类型也已经校验过
    print(f"目标：{plan.goal}")
    for t in plan.tasks:
        print(f"  Day {t.day}（{t.hours}h）{t.topic} → 产出：{t.deliverable}")
    print(f"总时长：{sum(t.hours for t in plan.tasks)} 小时")
    print("建议：", *plan.tips, sep="\n  - ")


# 【LangChain 接口 + 我写的】LangChain：with_structured_output；我写的：用抽取结果对比缺少哪些技能
def demo_extract_jd(model):
    print("\n=== 2. 信息抽取：解析招聘 JD ===")
    extractor = model.with_structured_output(JobPosting, method="function_calling")
    job = extractor.invoke(f"从下面的招聘信息中抽取结构化数据：\n{JD_TEXT}")
    print(job.model_dump_json(indent=2))

    # 拿到结构化数据后就能写程序逻辑了，例如对比自己的技能
    my_skills = {"python", "langchain", "fastapi"}
    missing = [s for s in job.must_have if not any(m in s.lower() for m in my_skills)]
    print(f"\n你还需要补的硬性技能：{missing}")


# 【LangChain 用法】with_structured_output(include_raw=True) 及它返回的 raw / parsed / parsing_error
def demo_include_raw(model):
    print("\n=== 3. include_raw：调试时查看模型原始输出 ===")
    extractor = model.with_structured_output(JobPosting, method="function_calling", include_raw=True)
    result = extractor.invoke("阿里招 Java 后端，杭州，3 年经验，要求 Spring Boot")
    # 返回 dict：raw 是原始 AIMessage，parsed 是解析结果，parsing_error 是解析异常
    print(f"原始 tool_calls：{result['raw'].tool_calls}")
    print(f"解析结果：{result['parsed']}")
    print(f"解析错误：{result['parsing_error']}")


# 【我写的】把上面的例子依次跑一遍
if __name__ == "__main__":
    model = get_chat_model(temperature=0)  # 抽取类任务用低温度，结果更稳定
    demo_study_plan(model)
    demo_extract_jd(model)
    demo_include_raw(model)
