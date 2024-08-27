from langchain_community.document_loaders import UnstructuredExcelLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# 文档向量化
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
# llm
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models import HuggingFaceLLM  # 使用 HuggingFace 的本地模型

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
# 短时记忆
from langchain.memory import ConversationBufferMemory
# 实体
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationEntityMemory
# 总结
from langchain.memory import ConversationSummaryMemory
# 在长对话里面查找、删除、添加对话
from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
# 异步添加消息
import asyncio

# 多轮对话内存缓冲区窗口
from langchain.memory import ConversationBufferWindowMemory

from tika import parser
import pickle
import os

local_llmodel_path = "E:\\AAAAWork\\python\\models\\gpt2"

# 从环境变量中获取API密钥，用于初始化 ChatTongyi 模型
api_key = os.getenv("KEY_TONGYI")
if not api_key:
    raise ValueError("API Key is not set. Please ensure that the 'KEY_TONGYI' environment variable is set.")


# 打印并设置Tika服务器的路径，使用本地运行的 Tika Server 解析文件
print(os.path.abspath('./template/tika-server-standard-2.6.0.jar'))
os.environ['TIKA_SERVER_JAR'] = 'http://localhost:9998/'


# 初始化 ChatTongyi 模型，设置文本生成的温度参数，温度越低生成的文本越接近输入
llm = ChatTongyi(
    dashscope_api_key=api_key,
    temperature=0  # 设置生成文本的倾向，值越小生成的文本越接近输入
)


class ChatDoc:
    def __init__(self, folder_path):
        """
        初始化 ChatDoc 类，接受文件夹路径作为输入。
        :param folder_path: 包含文档的文件夹路径
        """
        self.folder_path = folder_path  # 设置文档文件夹路径
        self.splitText = []  # 初始化一个列表，用于存储分割后的所有文件文本块

        # template: 这是实际的提示内容，包含了占位符。
        # 该模板告诉模型提取对话中的所有命名实体，并且以列表形式返回这些实体。
        entity_extraction_prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="从以下对话中提取所有命名实体:\n\n{history}\n\nUser: {input}\n\nEntities:",
        )
        entity_summarization_prompt = PromptTemplate(
            input_variables=["summary", "entity", "history", "input"],
            template=(
                "根据以下对话历史记录和用户输入，更新实体的摘要 '{entity}'.\n\n"
                "History:\n{history}\n\nUser: {input}\n\nCurrent Summary: {summary}\n\nUpdated Summary:"
            ),
        )

        # 初始化记忆实体概念清单
        self.memoryEntityMemory = ConversationEntityMemory(
            llm=llm,
            entity_extraction_prompt=entity_extraction_prompt,
            entity_summarization_prompt=entity_summarization_prompt,
            human_prefix="User",  # 定义人类发言的前缀
            ai_prefix="AI",       # 定义AI发言的前缀
        )
        # 总结
        # 初始化对话历史和摘要内存
        self.memorySummaryMemory = ConversationSummaryMemory(
            llm=llm,
            memory_key="summary",
            chat_memory=None  # 可以根据需要初始化一个BaseChatMessageHistory实例
        )

        # 初始化ChatMessageHistory 在长对话里面查找、删除、添加对话
        self.memoryChatMessageHistory = ChatMessageHistory()

        # 初始化对话窗口内存，设置窗口大小为最近 3 对（6 条）消息
        self.memoryBufferWindowMemory = ConversationBufferWindowMemory(k=5)

        # 添加短时记忆初始化
        self.memoryBufferMemory = ConversationBufferMemory(human_prefix="User", ai_prefix="AI")
        # 定义一个聊天提示模板，用于在对话中生成提示消息
        self.template = [
            ("system", "你是一个处理文档的秘书,\
             你会根据下面提供的上下文内容来继续回答问题,\
             你从不说自己是一个大模型或者AI助手.\
             \n上下文内容\n{context}\n"),
            ("human", "你好!\n"),
            ("ai", "你好!"),
            ("human", "{question}\n"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.template)  # 创建聊天提示模板

    def parse_folder_with_tika(self):
        """
        使用 Apache Tika 解析文件夹中的所有文件，并将内容传递给 splitSentences 方法处理。
        """
        for filename in os.listdir(self.folder_path):  # 遍历文件夹中的每个文件
            file_path = os.path.join(self.folder_path, filename)  # 获取每个文件的完整路径
            if os.path.isfile(file_path):  # 检查路径是否是文件
                try:
                    # 使用 Tika 解析文件内容
                    parsed = parser.from_file(file_path)
                    text = parsed.get('content')  # 提取文件内容
                    if text:
                        print(f"文件 {filename} 解析成功，内容长度：{len(text.strip())}")
                        self.splitSentences(text.strip())  # 将解析的文本分割并存储
                    else:
                        print(f"文件 {filename} 解析后内容为空")
                except Exception as e:
                    print(f"文件 {filename} 解析失败: {e}")  # 处理解析失败的情况

    def splitSentences(self, full_text):
        """
        将完整文本分割为较小的块，并将它们存储为 Document 对象，以便后续处理。
        :param full_text: 需要分割的完整文本
        """
        if full_text:
            # 使用 CharacterTextSplitter 以 160 字符为一块分割文本，每块有 20 字符的重叠部分
            text_splitter = CharacterTextSplitter(chunk_size=160, chunk_overlap=20)
            texts = text_splitter.split_text(full_text)  # 分割文本
            for text in texts:
                doc = Document(page_content=text)  # 将分割后的文本块存储为 Document 对象
                self.splitText.append(doc)  # 将 Document 对象添加到 splitText 列表中
            print(f"分割后的文本块数量: {len(self.splitText)}")

    def embeddingAndVectorDB(self, index_path="faiss_index"):
        """
        将分割后的文本进行嵌入并存储在向量数据库中，如果索引文件存在则加载，否则创建新的索引。
        :param index_path: 索引文件的路径（不包括文件扩展名）
        :return: FAISS 向量数据库对象
        """
        if not self.splitText:
            raise ValueError("分割后的文本块为空，无法进行嵌入处理")

        if os.path.exists(f"{index_path}.index"):
            print("加载现有的 FAISS 索引...")
            db = self.load_faiss_index(index_path)
        else:
            print("创建新的 FAISS 索引...")
            # 定义本地模型路径
            local_em_model_path = "E:\\AAAAWork\\python\\models\\bce-embedding-base_v1"  # 替换为你的本地模型路径

            # 使用 DashScopeEmbeddings 模型创建嵌入表示
            hf = HuggingFaceEmbeddings(
                model_name=local_em_model_path, # 本地模型路径或名称
                model_kwargs={"device": "cpu"},  # 如果你使用GPU，将"cpu"替换为"cuda"
                encode_kwargs={"normalize_embeddings": True},  # 可选参数，是否归一化嵌入
            )
            # 使用 FAISS 创建向量数据库，存储分割后的文本块
            db = FAISS.from_documents(documents=self.splitText, embedding=hf)
            # 保存索引和文档到磁盘
            self.save_faiss_index(db, index_path)

        return db  # 返回向量数据库对象

    def save_faiss_index(self, db, index_path="faiss_index"):
        """
        保存 FAISS 向量数据库到磁盘。
        :param db: FAISS 向量数据库对象
        :param index_path: 保存索引文件的路径（不包括文件扩展名）
        """
        db.save_local(index_path)
        with open(f"{index_path}_docs.pkl", "wb") as f:
            pickle.dump(self.splitText, f)
        print(f"FAISS 索引和文档已保存到 {index_path}")

    def load_faiss_index(self, index_path="faiss_index"):
        """
        从磁盘加载 FAISS 向量数据库。
        :param index_path: 索引文件的路径（不包括文件扩展名）
        :return: 加载的 FAISS 向量数据库对象
        """
        # 定义本地模型路径
        local_em_model_path = "E:\\AAAAWork\\python\\models\\bce-embedding-base_v1"  # 替换为你的本地模型路径

        # 使用 DashScopeEmbeddings 模型创建嵌入表示
        hf = HuggingFaceEmbeddings(
            model_name=local_em_model_path, # 本地模型路径或名称
            model_kwargs={"device": "cpu"},  # 如果你使用GPU，将"cpu"替换为"cuda"
            encode_kwargs={"normalize_embeddings": True},  # 可选参数，是否归一化嵌入
        )
        # 使用 FAISS 创建向量数据库，存储分割后的文本块
        db = FAISS.load_local(index_path, embedding=hf)
        with open(f"{index_path}_docs.pkl", "rb") as f:
            self.splitText = pickle.load(f)
        print(f"FAISS 索引和文档已从 {index_path} 加载")

        return db

    def askAndFindFiles(self, question):
        """
        根据用户的问题在文档中查找相关内容。
        :param question: 用户的查询问题
        :return: 相关文档块的列表
        """
        db = self.embeddingAndVectorDB()  # 获取向量化的文本块
        retriever = db.as_retriever()  # 创建检索器
        # 使用 LLMChainExtractor 作为压缩器，结合 ChatTongyi 模型进行上下文压缩检索
        compressor = LLMChainExtractor.from_llm(llm=llm)
        compressor_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        # 返回与问题相关的文档块列表
        return compressor_retriever.get_relevant_documents(query=question)

    def chatWithDoc(self, question):
        """
        与文档进行对话，通过解析文件夹、查找相关内容并生成回答。
        :param question: 用户的问题
        :return: 生成的回答文本
        """
        self.parse_folder_with_tika()  # 首先解析文件夹中的所有文件

        if not self.splitText:
            raise ValueError("没有从文档中获取到任何内容，无法进行对话")  # 如果没有解析到内容，抛出错误

        context = self.askAndFindFiles(question)  # 查找与问题相关的文本块
        _context = ""
        for i in context:
            if hasattr(i, 'page_content'):
                _context += i.page_content  # 汇总所有相关文本块的内容
            else:
                print(f"对象 {i} 没有 'page_content' 属性，内容: {i}")

        # 提取各类记忆
        short_term_memory = self.memoryBufferMemory.load_memory_variables({"input": question})[self.memoryBufferMemory.memory_key]
        entity_memory = self.memoryEntityMemory.load_memory_variables({"input": question})["entities"]
        summary_memory = self.memorySummaryMemory.load_memory_variables({"input": question})[self.memorySummaryMemory.memory_key]

        # 将提取的记忆内容与当前上下文合并，并显式标注记忆类型
        full_context = (
            f"Short-term Memory:\n{short_term_memory}\n\n"
            f"Conversation Summary:\n{summary_memory}\n\n"
            f"Relevant Documents:\n{_context}\n\n"
            f"Entity Memory:\nEntities: {entity_memory}\n"
        )
        # 格式化消息内容，生成系统、用户和 AI 的对话
        message = self.prompt.format_messages(context=full_context, question=question)

        # 回答
        response = llm.invoke(message)

        # 更新记忆内容
        # 添加人的问题的记忆
        self.memoryBufferMemory.chat_memory.add_user_message(question)
        self.memoryEntityMemory.chat_memory.add_user_message(question)
        self.memorySummaryMemory.chat_memory.add_user_message(question)
        # self.memoryChatMessageHistory.add_message(HumanMessage(content=question))
        self.memoryBufferWindowMemory.chat_memory.add_user_message(question)

        # 添加ai的回答的记忆
        self.memoryBufferMemory.chat_memory.add_ai_message(response.content)
        self.memoryEntityMemory.chat_memory.add_ai_message(response.content)
        self.memorySummaryMemory.chat_memory.add_ai_message(response.content)
        # self.memoryChatMessageHistory.add_message(AIMessage(content=response.content))
        self.memoryBufferWindowMemory.chat_memory.add_ai_message(response.content)
        # 异步添加
        async def add_async_messages():
            await self.memoryChatMessageHistory.aadd_messages([
                HumanMessage(content=question),
                AIMessage(content=response.content)
            ])
            
            # await self.memoryBufferWindowMemory.chat_memory.aadd_messages([
            #     HumanMessage(content=question),
            #     AIMessage(content=response.content)
            # ])

        asyncio.run(add_async_messages())

        # 加载实体记忆
        memory_variables = self.memoryEntityMemory.load_memory_variables({"input": question})
        # print("Memory Variables after first interaction:", memory_variables)

        # 保存对话的上下文和生成的实体
        self.memoryEntityMemory.save_context({"input":question},{"output":response.content})
        
        self.memorySummaryMemory.save_context({"input":question},{"output":response.content})

        # 查看对话总结
        summary_memory_variables = self.memorySummaryMemory.load_memory_variables({"input": question})
        # print("Summary Memory after third interaction:", summary_memory_variables)

        # 假设响应包含 token 使用情况的信息
        if hasattr(response, 'usage'):
            tokens_used = response.usage.get('total_tokens', 'N/A')
            tokens_remaining = response.usage.get('tokens_remaining', 'N/A')  # 如果模型支持显示剩余 tokens
            print(f"消耗的 Tokens: {tokens_used}")
            print(f"剩余的 Tokens: {tokens_remaining}")

        return response

        # # 使用 LLM 生成回答
        # return llm.invoke(message)


# 使用示例：处理文件夹中的所有文件并与文档进行对话
folder_path = os.path.abspath('./docx_data')  # 获取文档文件夹的绝对路径
chat_doc = ChatDoc(folder_path)

loop = True
while(loop):
    str_question = input("user:")
    if str_question == 'q':
        break
    response = chat_doc.chatWithDoc(str_question)  # 提出一个问题，并生成回答

    # print("====================回答====================")  # 打印回答
    print('ai:', response.content)  # 打印回答

# 短时记忆
print('chat_doc.memoryBufferMemory:',chat_doc.memoryBufferMemory.load_memory_variables({}))
# 查看对话历史和实体 摘要
final_memory = chat_doc.memoryEntityMemory.load_memory_variables({"input": ""})
print("chat_doc.memoryEntityMemory:", final_memory)

# 异步获取所有消息
async def print_async_messages():
    async_messages = await chat_doc.memoryChatMessageHistory.aget_messages()
    print("\nAsync memoryChatMessageHistory:")
    for msg in async_messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")

asyncio.run(print_async_messages())

# 由于窗口大小设置为 3 对消息（6 条），只保留最新的三条
# 查看历史记录（字符串形式）
history_as_str = chat_doc.memoryBufferWindowMemory.buffer_as_str
print("History as String:")
print(history_as_str)

# 查看历史记录（消息列表形式）
history_as_messages = chat_doc.memoryBufferWindowMemory.buffer_as_messages
print("\nHistory as Messages:")
for message in history_as_messages:
    role = "User" if isinstance(message, HumanMessage) else "AI"
    print(f"{role}: {message.content}")

# 加载内存变量
memory_variables = chat_doc.memoryBufferWindowMemory.load_memory_variables({})
print("\nMemory Variables:")
print(memory_variables)