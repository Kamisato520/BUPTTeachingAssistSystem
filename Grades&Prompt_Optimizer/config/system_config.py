class SystemConfig:
    def __init__(self):
        self.model_configs = {
            "qwen-plus": {
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "gpt-4": {
                "max_tokens": 4096,
                "temperature": 0.8
            }
        }
        
    def get_model_config(self, model_name):
        return self.model_configs.get(model_name, {}) 