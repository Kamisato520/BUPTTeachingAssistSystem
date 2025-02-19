from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
"""
在需要使用prompt的地方
optimizer = PromptOptimizer(model)
optimization_result = await optimizer.optimize_prompt(original_prompt)
optimized_prompt = optimization_result['optimized_prompt']
使用优化后的prompt调用模型
response = model.generate_response(optimized_prompt)
"""
class PromptOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimization_prompt_template = """
请对以下提示词进行优化，使其更加清晰、具体和结构化。
要求：
1. 明确任务目标和期望输出
2. 添加必要的上下文信息
3. 提供具体的输出格式要求
4. 设置适当的约束条件
请考虑：
        1. 评分标准的明确性
        2. 关键点的覆盖程度
        3. 表述的专业性
原始提示词：
{original_prompt}

请按照以下JSON格式返回优化结果：
{
    "optimized_prompt": "优化后的提示词",
    "improvements": ["改进点1", "改进点2", ...],
    "reasoning": "优化理由"
}
"""

    async def optimize_prompt(self, original_prompt: str) -> Dict[str, Any]:
        """
        优化输入的prompt
        
        Args:
            original_prompt: 原始提示词
            
        Returns:
            Dict包含优化后的提示词和优化说明
        """
        try:
            # 构建优化请求
            optimization_request = self.optimization_prompt_template.format(
                original_prompt=original_prompt
            )
            
            # 调用模型进行优化
            optimization_response = self.model.generate_response(optimization_request)
            
            # 解析优化结果
            try:
                result = eval(optimization_response)  # 将字符串转换为字典
                logger.info(f"Prompt优化成功: {result['improvements']}")
                return result
            except Exception as e:
                logger.error(f"优化结果解析失败: {e}")
                return {
                    "optimized_prompt": original_prompt,
                    "improvements": ["优化失败，使用原始提示词"],
                    "reasoning": f"解析错误: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Prompt优化过程出错: {e}")
            return {
                "optimized_prompt": original_prompt,
                "improvements": ["优化过程出错，使用原始提示词"],
                "reasoning": str(e)
            }

    def get_task_specific_template(self, task_type: str) -> str:
        """
        获取特定任务类型的提示词模板
        
        Args:
            task_type: 任务类型（如'grading', 'analysis'等）
            
        Returns:
            str: 任务特定的提示词模板
        """
        templates = {
            'grading': """
请对以下答案进行评分和分析：
问题：{question}
答案：{answer}
参考标准：{criteria}

请提供：
1. 分数（0-100分）
2. 详细的评分理由
3. 具体的改进建议
            """,
            'analysis': """
请分析以下数据并生成报告：
数据：{data}

要求提供：
1. 关键统计指标
2. 数据分布分析
3. 重要发现和建议
            """
        }
        return templates.get(task_type, "{prompt}") 