import uuid

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 设置 Embedding Function
embeddings = OpenAIEmbeddings(
    openai_api_key="sk-WKGLTTbN0d583D64555bT3BlBkFJ50E2848daCF9409AAe18",
    openai_api_base="https://cfcus02.opapi.win/v1"
)

# 创建或连接到 Chroma 数据库
vectorstore = Chroma(
    persist_directory="../knowledge_base_demo.db",
    embedding_function=embeddings
)

# 示例文档数据
documents = [
    {"content": "供需关系是经济学的核心概念之一，描述了市场中商品供给与需求之间的关系。"},
    {"content": "价格弹性是指需求量对价格变化的敏感程度，可以通过公式计算：价格弹性 = (需求量变化百分比) / (价格变化百分比)。"},
    {"content": "消费者剩余是指消费者愿意支付的最高价格与实际支付价格之间的差额。"},
    {"content": "边际效用递减法则表示，随着消费者消费某种商品的数量增加，所获得的额外效用会逐渐减少。"},
    {"content": "市场均衡是供给曲线和需求曲线的交点，表示市场中供给与需求相等的状态。"},
]

# # 添加文档到 Chroma 数据库
# for doc in documents:
#     vectorstore.add_texts(
#         texts=[doc["content"]],
#         metadatas=[{"id": str(uuid.uuid4())}]
#     )
#
# print("Chroma 数据库已初始化并自动持久化。")
docs = vectorstore.similarity_search("", k=1000)  # 通过空查询获取所有记录
for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
