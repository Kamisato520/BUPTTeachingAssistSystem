import unittest
import logging
from core.model import GPT4oModel
from core.knowledge_base import KnowledgeBase
from core.grading import GradingSystem

logger = logging.getLogger(__name__)

class TestGradingSystem(unittest.TestCase):
    def setUp(self):
        """测试前初始化"""
        self.model = GPT4oModel()
        self.kb = KnowledgeBase(self.model)
        self.grader = GradingSystem(self.model, self.kb)

    def test_grade_with_feedback(self):
        """测试评分和反馈功能"""
        # 测试输入
        question = "请解释机器学习的概念。"
        answer = "机器学习是人工智能的一个分支，通过数据训练算法。"
        
        logger.info("=== 评分测试开始 ===")
        logger.info(f"测试问题: {question}")
        logger.info(f"测试答案: {answer}")
        
        # 添加背景知识
        background = "机器学习是人工智能的核心领域之一，主要研究如何通过计算的手段，利用经验来改善系统自身的性能。"
        self.kb.add_background_knowledge(question, background)
        logger.info(f"添加的背景知识: {background}")
        
        # 获取评分结果
        result = self.grader.grade_with_feedback(question, answer)
        
        # 输出详细结果
        logger.info("\n=== 评分结果 ===")
        logger.info(f"分数: {result['score']}")
        logger.info(f"反馈: {result['feedback']}")
        logger.info(f"改进建议: {result['improvement']}")
        logger.info(f"评分理由: {result['rationale']}")
        
        # 验证结果格式
        self.assertIn('score', result)
        self.assertIn('feedback', result)
        self.assertIn('improvement', result)
        self.assertIn('rationale', result)
        
        # 验证分数范围
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)

if __name__ == '__main__':
    unittest.main() 