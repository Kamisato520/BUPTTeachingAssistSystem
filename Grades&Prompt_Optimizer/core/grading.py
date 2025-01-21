import json
from typing import Dict, Any
from core.prompt_optimizer import PromptOptimizer

class GradingSystem:
    def __init__(self, model, knowledge_base):
        self.model = model
        self.knowledge_base = knowledge_base
        self.scoring_criteria = self._load_criteria()
        self.prompt_optimizer = PromptOptimizer(model)#prompt优化器

    def _load_criteria(self):
        # 加载评分标准
        return {
            "content": 0.4,
            "logic": 0.4,
            "clarity": 0.2
        }

    def grade_with_feedback(self, question: str, answer: str) -> Dict[str, Any]:
        """
        评分并提供反馈
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            Dict包含分数、反馈和改进建议
        """
        if not question or not answer:
            raise ValueError("问题和答案不能为空")
            
        try:
            score_result = self.grade_subjective_with_rag(question, answer)
            
            # 使用更结构化的prompt
            feedback_prompt = (
                "请按照以下格式提供答案评价：\n"
                "1. 内容准确性\n"
                "2. 逻辑性\n"
                "3. 表达清晰度\n"
                f"问题：{question}\n"
                f"答案：{answer}"
            )
            
            feedback = self.model.generate_response(feedback_prompt)
            
            improvement_prompt = (
                "请提供具体的改进建议，包括：\n"
                "1. 需要补充的关键点\n"
                "2. 逻辑结构改进\n"
                "3. 表达方式优化\n"
                f"答案：{answer}"
            )
            
            improvement = self.model.generate_response(improvement_prompt)
            
            return {
                "score": score_result["score"],
                "feedback": feedback,
                "improvement": improvement,
                "rationale": score_result["rationale"]
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "feedback": "评分过程发生错误",
                "improvement": "无法生成改进建议",
                "rationale": str(e)
            }

    async def grade_subjective_with_rag(self, question: str, student_answer: str) -> Dict[str, Any]:
        """使用优化后的prompt进行评分"""
        context_documents = self.knowledge_base.search_knowledge(question)
        context_text = "\n".join([doc['text'] for doc in context_documents])
        
        # 构建原始prompt
        original_prompt = (
            f"基于以下信息评分（0-1分）并提供理由：\n"
            f"问题：{question}\n"
            f"答案：{student_answer}\n"
            f"参考资料：{context_text}"
        )
        
        # 优化prompt
        optimization_result = await self.prompt_optimizer.optimize_prompt(original_prompt)
        optimized_prompt = optimization_result['optimized_prompt']
        
        # 使用优化后的prompt调用模型
        response = self.model.generate_response(optimized_prompt)
        
        try:
            result = json.loads(response)
            return {
                "score": float(result.get("score", 0.0)),
                "rationale": result.get("rationale", "未提供评分理由"),
                "prompt_improvements": optimization_result.get("improvements", [])
            }
        except:
            return {
                "score": 0.0,
                "rationale": "评分解析失败",
                "prompt_improvements": ["prompt优化失败"]
            }