import pytest
import json
import os

@pytest.fixture
def mock_api_config():
    return {
        "chat_completion": {
            "url": "http://mock-api.test/chat",
            "headers": {"Content-Type": "application/json"},
            "auth_key": "test-key"
        },
        "vectorize": {
            "url": "http://mock-api.test/vectorize",
            "headers": {"Content-Type": "application/json"},
            "auth_key": "test-key"
        }
    }

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    # 创建临时配置文件
    config_path = tmp_path / "test_api_config.json"
    with open(config_path, 'w') as f:
        json.dump(mock_api_config(), f)
    
    # 设置环境变量
    os.environ['API_CONFIG_PATH'] = str(config_path)
    return str(config_path) 