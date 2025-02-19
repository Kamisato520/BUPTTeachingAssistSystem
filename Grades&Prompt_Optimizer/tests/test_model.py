import unittest
import logging
from core.model import GPT4oModel
from unittest.mock import patch

logger = logging.getLogger(__name__)

class TestGPT4oModel(unittest.TestCase):
    def setUp(self):
        """测试前初始化"""
        self.model = GPT4oModel()

    def test_chat_completion(self):
        """测试聊天补全功能"""
        # 测试输入
        messages = [{"role": "user", "content": "你是谁？"}]
        logger.info(f"测试输入: {messages}")
        
        # 调用API
        response = self.model.chat_completion(messages)
        logger.info(f"API响应: {response}")
        
        # 验证结果
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        logger.info(f"响应长度: {len(response)} 字符")

    @patch('core.model.GPT4oModel._call_api')
    def test_vectorize_text(self, mock_call_api):
        """测试文本向量化"""
        # 模拟API返回值
        mock_response = {"vector": [0.1, 0.2, 0.3]}
        mock_call_api.return_value = mock_response
        
        # 测试输入
        test_text = "测试文本"
        logger.info(f"测试输入文本: {test_text}")
        
        # 调用API
        vector = self.model.vectorize_text(test_text)
        logger.info(f"生成的向量: {vector}")
        
        # 验证结果
        self.assertIsInstance(vector, list)
        logger.info(f"向量维度: {len(vector)}")

if __name__ == '__main__':
    unittest.main() 