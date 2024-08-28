from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型
model = SentenceTransformer("E:\\AAAAWork\\python\\models\\bce-embedding-base_v1")

# 读取txt文档，句子列表
with open(r"E:\AAAAWork\python\LLM_RAG\huggingface\test.txt", 'r', encoding='utf-8') as f:
    sentences = f.read().splitlines()

# 将所有句子进行向量化
embeddings = model.encode(sentences)

# 打印向量化后的内容
print("向量化后的内容：")
for i, embedding in enumerate(embeddings):
    print(f"Sentence {i+1}: {sentences[i]}")
    print(f"Embedding: {embedding}")
    print()  # 空行分隔各个句子的嵌入
