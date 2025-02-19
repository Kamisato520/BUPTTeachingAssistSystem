from core.model import GPT4oModel
from chromadb import Client, Settings
import chromadb
import os

class KnowledgeBase:
    def __init__(self, model: GPT4oModel):
        self.model = model
        self.categories = set()
        self.vector_store = []
        self.dimension = 1536  # text-embedding-ada-002的维度
        
        # 使用相对路径设置ChromaDB
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(os.path.dirname(current_dir), "knowledge_base_demo")
        
        # 初始化ChromaDB客户端
        self.client = Client(Settings(
            persist_directory=db_path,
            anonymized_telemetry=False
        ))
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.texts = []  # 存储原始文本的映射

    async def add_to_knowledge_base(self, document, metadata=None):
        """添加文档到知识库"""
        if not metadata:
            metadata = {}
            
        # 生成文档的embedding
        embedding = self.model.get_embedding(document)
        
        # 生成唯一ID
        doc_id = str(len(self.collection.get()['ids']) + 1)
        
        # 添加到ChromaDB
        self.collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def search_knowledge(self, query, top_k=5):
        """
        检索与问题相关的知识。
        """
        return self.model.search_knowledge(query, top_k)

    def add_background_knowledge(self, question, background):
        """添加背景知识到知识库"""
        # 生成embedding
        embedding = self.model.get_embedding(background)
        
        # 准备数据
        data = [
            [background],           # text field
            [embedding.tolist()]    # embedding field
        ]
        
        # 插入数据
        self.collection.add(
            documents=[background],
            embeddings=[embedding.tolist()],
            metadatas=[{"category": "background"}],
            ids=[str(len(self.collection.get()['ids']) + 1)]
        )
        self.texts.append(background)

    async def retrieve_relevant_knowledge(self, query, k=3):
        """检索相关知识"""
        query_embedding = self.model.get_embedding(query)
        
        # 使用ChromaDB进行相似度搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        return results['documents'][0]  # 返回最相关的文档列表

    def add_to_knowledge_base(self, document, metadata=None):
        """支持更多文档格式和索引方式"""
        vector = self.model.vectorize_text(document)
        
        # 添加文档分类
        if metadata and "category" in metadata:
            self.categories.add(metadata["category"])
            
        self.vector_store.add_texts([document], [metadata])

    def __del__(self):
        """清理连接"""
        pass

    async def retrieve_and_augment(self, query, k=3):
        """RAG的核心方法：检索并增强"""
        # 1. 检索相关知识
        relevant_texts = await self.retrieve_relevant_knowledge(query, k)
        
        # 2. 构建增强的上下文
        context = self._build_augmented_context(query, relevant_texts)
        
        # 3. 使用增强的上下文生成回答
        augmented_response = await self.model.generate_completion(context)
        
        return {
            'response': augmented_response,
            'retrieved_contexts': relevant_texts,
            'augmented_context': context
        }

    def _build_augmented_context(self, query, relevant_texts):
        """构建增强的上下文"""
        context = "基于以下参考信息回答问题：\n\n"
        
        # 添加检索到的相关文本
        for i, text in enumerate(relevant_texts, 1):
            context += f"参考{i}：{text}\n\n"
        
        # 添加原始查询
        context += f"问题：{query}\n"
        context += "请基于上述参考信息，生成准确和全面的回答。"
        
        return context

    async def grade_with_rag(self, question, student_answer, grading_criteria):
        """使用RAG进行评分"""
        # 1. 检索相关的参考答案和评分标准
        relevant_knowledge = await self.retrieve_relevant_knowledge(question, k=3)
        
        # 2. 构建评分上下文
        grading_context = self._build_grading_context(
            question,
            student_answer,
            relevant_knowledge,
            grading_criteria
        )
        
        # 3. 使用增强的上下文进行评分
        grading_result = await self.model.generate_completion(grading_context)
        
        return {
            'score': self._extract_score(grading_result),
            'feedback': grading_result,
            'reference_materials': relevant_knowledge
        }

    def _build_grading_context(self, question, student_answer, relevant_knowledge, grading_criteria):
        """构建评分上下文"""
        context = f"""请基于以下信息对学生答案进行评分：

问题：
{question}

参考知识：
"""
        # 添加检索到的相关知识
        for i, knowledge in enumerate(relevant_knowledge, 1):
            context += f"{i}. {knowledge}\n"

        context += f"""
评分标准：
{grading_criteria}

学生答案：
{student_answer}

请提供：
1. 分数（满分100分）
2. 详细的评分理由
3. 改进建议
"""
        return context

    def _extract_score(self, grading_result):
        """从评分结果中提取分数"""
        try:
            # 简单实现，实际应该使用更复杂的解析逻辑
            import re
            score_match = re.search(r'分数[：:]\s*(\d+)', grading_result)
            if score_match:
                return int(score_match.group(1))
            return None
        except Exception:
            return None

    async def add_to_knowledge_base(self, document, metadata=None):
        """添加文档到知识库，支持批量处理"""
        # 生成embedding
        embedding = self.model.get_embedding(document)
        
        # 准备数据
        data = [
            [document],           # text field
            [embedding.tolist()]  # embedding field
        ]
        
        # 插入数据
        self.collection.add(
            documents=[document],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[str(len(self.collection.get()['ids']) + 1)]
        )
        self.texts.append(document)
        
        # 添加到向量存储
        if metadata:
            # 可以存储额外的元数据
            pass
            
        # 刷新确保可搜索
        self.collection.flush() 