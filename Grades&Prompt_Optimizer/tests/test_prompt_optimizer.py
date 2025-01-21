import unittest
import logging
from core.prompt_optimizer import PromptOptimizer
from core.model import GPT4oModel

logger = logging.getLogger(__name__)

class TestPromptOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = GPT4oModel()
        self.optimizer = PromptOptimizer(self.model)

    async def test_optimize_prompt(self):
        """测试prompt优化功能"""
        original_prompt = "解释机器学习"
        
        logger.info(f"原始prompt: {original_prompt}")
        
        result = await self.optimizer.optimize_prompt(original_prompt)
        
        logger.info("优化结果:")
        logger.info(f"优化后prompt: {result['optimized_prompt']}")
        logger.info(f"改进点: {result['improvements']}")
        logger.info(f"优化理由: {result['reasoning']}")
        
        self.assertIn('optimized_prompt', result)
        self.assertIn('improvements', result)
        self.assertIn('reasoning', result)
        
        # 验证优化后的prompt更长更详细
        self.assertGreater(len(result['optimized_prompt']), len(original_prompt))

if __name__ == '__main__':
    unittest.main() 