import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
import logging
from tqdm import tqdm

class DataImporter:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.logger = logging.getLogger(__name__)
        
    async def import_from_files(self, directory: str, file_types: List[str] = None):
        """从指定目录导入文件到ChromaDB"""
        if file_types is None:
            file_types = ['.txt', '.pdf', '.docx', '.json', '.csv']
            
        path = Path(directory)
        files = []
        for file_type in file_types:
            files.extend(path.glob(f'**/*{file_type}'))
            
        for file in tqdm(files, desc="导入文件"):
            try:
                if file.suffix == '.json':
                    await self.import_json(str(file))
                elif file.suffix == '.csv':
                    await self.import_csv(str(file))
                elif file.suffix in ['.txt', '.pdf', '.docx']:
                    await self.import_document(str(file))
            except Exception as e:
                self.logger.error(f"导入文件 {file} 失败: {str(e)}")

    async def import_json(self, file_path: str):
        """导入JSON格式的知识库数据到ChromaDB"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for item in tqdm(data, desc="导入JSON数据"):
                await self._process_json_item(item)
        elif isinstance(data, dict):
            await self._process_json_item(data)

    async def _process_json_item(self, item: Dict):
        """处理单个JSON数据项并添加到ChromaDB"""
        if isinstance(item, dict):
            content = item.get('content')
            metadata = item.get('metadata', {})
            if content:
                await self.kb.add_to_knowledge_base(content, metadata)

    async def import_csv(self, file_path: str, content_column: str = 'content'):
        """导入CSV格式的知识库数据"""
        df = pd.read_csv(file_path)
        if content_column not in df.columns:
            raise ValueError(f"CSV文件中未找到内容列: {content_column}")
            
        for _, row in tqdm(df.iterrows(), desc="导入CSV数据", total=len(df)):
            content = row[content_column]
            # 将其他列作为metadata
            metadata = row.drop(content_column).to_dict()
            await self.kb.add_to_knowledge_base(content, metadata)

    async def import_document(self, file_path: str):
        """导入文档文件（TXT、PDF、DOCX等）"""
        from core.file_processor import FileProcessor
        
        content = FileProcessor.extract_text_from_file(file_path)
        if content:
            metadata = {
                'source': file_path,
                'type': Path(file_path).suffix[1:]
            }
            await self.kb.add_to_knowledge_base(content, metadata)

    async def import_from_database(self, connection_string: str, query: str):
        """从数据库导入数据"""
        import sqlalchemy as sa
        
        engine = sa.create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(sa.text(query))
            for row in tqdm(result, desc="从数据库导入"):
                content = str(row[0])  # 假设第一列是内容
                metadata = {
                    'source': 'database',
                    'query': query,
                    # 可以添加其他列作为metadata
                }
                await self.kb.add_to_knowledge_base(content, metadata)

    async def import_from_api(self, api_url: str, headers: Dict = None):
        """从API导入数据"""
        import aiohttp
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(api_url) as response:
                data = await response.json()
                
                if isinstance(data, list):
                    for item in tqdm(data, desc="从API导入"):
                        await self._process_api_item(item)
                elif isinstance(data, dict):
                    await self._process_api_item(data)

    async def _process_api_item(self, item: Dict):
        """处理从API获取的单个数据项"""
        if isinstance(item, dict):
            content = item.get('content')
            metadata = {
                'source': 'api',
                'id': item.get('id'),
                'timestamp': item.get('timestamp')
            }
            if content:
                await self.kb.add_to_knowledge_base(content, metadata) 