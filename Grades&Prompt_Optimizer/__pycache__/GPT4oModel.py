import os
import requests
import json

class GPT4oModel:
    def __init__(self, config_path='api_editor.json'):
        """
        初始化 GPT4o 模型，加载配置文件。
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在，请检查路径。")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 加载配置或提供默认值
        self.generate_config = config.get("generate", {})
        self.vectorize_config = config.get("vectorize", {})
        self.chat_config = config.get("chat_completion", {
            "url": "",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "auth_key": ""
        })
        # 打印加载的配置文件内容
        print(f"加载的配置文件内容：{config}")

        # 检查 chat_completion 配置的有效性
        if not self.chat_config.get("url"):
            raise ValueError("chat_completion 配置中的 url 字段不能为空，请检查配置文件。")

        # 打印加载的内容
        print(f"配置文件内容：{config}")

        self.model_name = "qwen-plus"  # 默认模型

    def _call_api(self, api_config, payload):
        """
        通用 API 调用方法。
        """
        url = api_config.get("url", "")
        method = api_config.get("method", "POST").upper()
        headers = api_config.get("headers", {})
        auth_key = api_config.get("auth_key", "")
        # 打印 URL 和请求体，调试用
        print(f"调用的 URL: {url}")
        print(f"请求体: {payload}")
        headers["Authorization"] = f"Bearer {auth_key}"

        if method == "POST":
            response = requests.post(url, json=payload, headers=headers)
        elif method == "GET":
            response = requests.get(url, params=payload, headers=headers)
        else:
            raise ValueError(f"不支持的 HTTP 方法：{method}")

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"API调用失败，状态码：{response.status_code}，错误信息：{response.text}")

    def generate_response(self, prompt):
        """
        调用 generate 接口生成响应。
        """
        payload = {"prompt": prompt}
        response = self._call_api(self.generate_config, payload)
        return response.get("response", "")

    def vectorize_text(self, text):
        """
        调用 vectorize 接口获取文本向量。
        """
        payload = {"text": text}
        response = self._call_api(self.vectorize_config, payload)
        return response.get("vector", [])

    def chat_completion(self, messages, model=None):
        """
        调用 Chat Completion 接口生成响应。
        """
        model = model or self.model_name
        payload = {"model": model, "messages": messages}
        response = self._call_api(self.chat_config, payload)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError("API响应格式错误，无法解析内容。")

if __name__ == "__main__":
    try:
        model = GPT4oModel(config_path="api_editor.json")

        # 测试 Chat Completion 功能
        messages = [{"role": "user", "content": "你是谁？"}]
        chat_response = model.chat_completion(messages)
        print("Chat Completion 功能测试结果：")
        print(chat_response)
    except Exception as e:
        print(f"Chat Completion 测试失败：{e}")
