from typing import List, Dict, Any, Optional
import json
import requests
from config.system_config import SystemConfig
from openai import OpenAI

class GPT4oModel:
    def __init__(self, config_path: str = 'config/api_config.json'):
        try:
            # 加载系统配置
            self.system_config = SystemConfig()
            self.model_name = "qwen-plus"
            
            # 加载 API 配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self.api_config = json.load(f)
                
            # 验证配置
            if not self.api_config.get("chat_completion", {}).get("url"):
                raise ValueError("API配置缺少必要的URL")
                
            self.client = OpenAI()
                
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件格式错误: {config_path}")

    def _call_api(self, config_key, payload):
        config = self.api_config.get(config_key, {})
        url = config.get('url')
        headers = config.get('headers', {})
        headers['Authorization'] = f"Bearer {config.get('auth_key', '')}"
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"API调用失败: {response.text}")

    def generate_response(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages)

    def vectorize_text(self, text):
        payload = {"text": text}
        result = self._call_api("vectorize", payload)
        return result.get('vector', [])

    def chat_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        调用聊天补全API
        
        Args:
            messages: 消息列表
            model: 可选的模型名称
            
        Returns:
            str: 模型响应的文本
            
        Raises:
            ValueError: API调用失败时抛出
        """
        model = model or self.model_name
        config = self.system_config.get_model_config(model)
        
        if not messages:
            raise ValueError("消息列表不能为空")
        
        payload = {
            "model": model,
            "messages": messages,
            **config
        }
        
        try:
            response = self._call_api("chat_completion", payload)
            return response["choices"][0]["message"]["content"]
        except KeyError:
            raise ValueError("API响应格式错误")

    def search_knowledge(self, query, top_k=5):
        # 简单实现，实际应该使用向量相似度搜索
        return [{"text": "示例知识", "metadata": {}}]

    def get_embedding(self, text):
        # 使用text-embedding-ada-002生成向量
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
        
