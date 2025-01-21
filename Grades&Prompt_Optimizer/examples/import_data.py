import asyncio
from core.knowledge_base import KnowledgeBase
from core.model import GPT4oModel
from core.data_importer import DataImporter
import logging

async def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化组件
    model = GPT4oModel()
    kb = KnowledgeBase(model)
    importer = DataImporter(kb)
    
    # 从不同来源导入数据
    
    # 1. 从文件导入
    await importer.import_from_files(
        directory="./data",
        file_types=['.json', '.csv', '.txt']
    )
    
    # 2. 从数据库导入
    await importer.import_from_database(
        connection_string="postgresql://user:pass@localhost/db",
        query="SELECT content, metadata FROM knowledge_table"
    )
    
    # 3. 从API导入
    await importer.import_from_api(
        api_url="https://api.example.com/knowledge",
        headers={"Authorization": "Bearer token"}
    )

if __name__ == "__main__":
    asyncio.run(main()) 