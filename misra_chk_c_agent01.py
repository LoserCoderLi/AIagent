import os
import time
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatTongyi
# from langchain.agents import load_tools, tool, create_react_agent
from langchain.agents import tool, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate

# 环境变量中的API密钥
api_key = os.getenv("TONGYI_API_KEY")

# LLM 配置
llm = ChatTongyi(
    dashscope_api_key=api_key,
    model="qwen-plus",
    temperature=0  # 0-1，越小越倾向于输入一致
)

# 引用预制工具
tools = load_tools(["llm-math"], llm)


# 定义自己的工具
@tool
def search_misra(query: str) -> str:
    """当需要了解MISRA C2012标准的相关信息，你可以调用此方法，
    你调用此工具需要输入信息必须包含违反的MISRA C 2012规则编码 , 违规内容的文字描述 以及 被检测的代码，
    文字描述应该包含你想到的可能违反的MISRA C 2012的规范细则，
    被检测代码应包含上下至少各三行代码以及相关变量的定义，而不是只有一行，
    规则编码 ， 违规内容的文字描述 ， 被检测的代码 三者缺一不可！
    然后返回MISRA C2012中对应的规范内容！"""
    rep = invoke_misra(query)
    return "MISRA C2012是一种编程规范，用于描述C语言编程语言的编程规范," + rep


@tool(return_direct=True)
def write_code(code: str) -> str:
    """
    将整理的结果输出为一个文件
    """
    filename = "static/support_c"
    full_file = filename + "_ai_chk.txt"
    with open(full_file, "w", encoding="utf-8") as file:
        file.write(code)
    print(full_file + "已生成")
    return code


def invoke_misra(query: str) -> str:
    print("正在调用MISRA C2012规范......")
    print("输入:" + query)

    # 配置模型
    model_name = "maidalun1020/bce-embedding-base_v1"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_emb = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    fs = LocalFileStore("cache")  # 本地缓存地址
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
        hf_emb, fs, namespace=hf_emb.model_name
    )
    vectorstore = FAISS.load_local("cache", cache_embeddings, allow_dangerous_deserialization=True)

    # 创建检索器
    retriever = vectorstore.as_retriever()

    # 上下文压缩
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    misra_template = [
        ("system",
         "你是一个精通C语言的C语言专家，在这个阶段你的主要目标是根据输入的内容匹配MISRA C2012标准的相关信息，输入内容{question}通常包含着违反的具体规则和错误代码,{context}通常包含着最相近的MISRA C2012细则，你需要根据输入的规则查阅MISRA C2012规范, 确定违反的规范细则是否准确，并将准确的细则和输入的代码一并返回"),
        ("human", "我会给你部分C语言代码和分析结果以及我认为它具体违反了MISRA C2012的哪些规范，你来帮我检查并纠正\n"),
        ("ai", "没问题!"),
        ("human", "{question}"),
    ]
    misra_prompt = ChatPromptTemplate.from_messages(misra_template)
    context = compressor_retriever.get_relevant_documents(query)
    print("********MISRA C2012 向量数据库命中*********")
    print(context)
    _context = "".join(i.page_content for i in context)
    message = misra_prompt.format_messages(context=_context, question=query)
    print("********MISRA C2012过程模板*********")
    print(misra_prompt)
    print("********MISRA C2012 命中细则*********")
    print(_context)
    rep = llm.invoke(message)
    print("********MISRA C2012过程响应*********")
    print(rep)
    print("***********************************")
    return rep.content


# 注册工具
tools.append(search_misra)
tools.append(write_code)

# Prompt 模板
template = '''尽力用中文回答下面的问题。你可以使用下面的工具:

{tools}

使用下面的格式:

Question: 你必须回答的问题
Thought: 你应该经常考虑该做什么，如果使用工具无法获得准确的结果，你应该终止操作，返回"陛下，臣妾做不到啊"这句话
Action: 要采取的行动，应该是[{tool_names}]中的一个
Action Input: 动作的输入
Observation: 行动的结果
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: 我知道最终的答案了
Final Answer: 原始输入问题的最终答案

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 定义 agent
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行代码并记录时间
start_time = time.time()
with open("static/unsupport_c.txt", "r", encoding="utf-8") as f:
    code = f.read()

rep = agent_executor.invoke({
                                "input": code + "这段代码哪里不符合MISRA C2012规范？在源代码上对不符合规范的地方做出标注，并将正确的C语言代码输出为本地的文件保存，注意，你输出的本地文件应该只包含C语言的代码和标注代码问题的注释。"})
print(rep["output"])
end_time = time.time()
execution_time = end_time - start_time
print(f"程序执行时间: {execution_time:.2f} 秒")
