import sys
import logging
from pathlib import Path
from core.model import GPT4oModel
from core.knowledge_base import KnowledgeBase
from core.grading import GradingSystem
from plugins.score_analysis import ScoreAnalysis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_system():
    """初始化系统组件"""
    try:
        # 检查配置文件
        config_path = Path("config/api_config.json")
        if not config_path.exists():
            raise FileNotFoundError("找不到API配置文件")
            
        # 初始化组件
        model = GPT4oModel(str(config_path))
        kb = KnowledgeBase(model)
        grader = GradingSystem(model, kb)
        
        return model, kb, grader
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        sys.exit(1)

def main():
    try:
        # 初始化系统
        model, kb, grader = init_system()
        
        # 测试评分
        question = "请解释机器学习的概念。"
        answer = "机器学习是人工智能的一个分支，通过数据训练算法。"
        
        # 添加背景知识
        kb.add_background_knowledge(
            question,
            "机器学习是人工智能的核心领域之一，主要研究如何通过计算的手段，"
            "利用经验来改善系统自身的性能。"
        )
        
        # 评分并获取反馈
        result = grader.grade_with_feedback(question, answer)
        logger.info("评分结果：%s", result)
        
    except Exception as e:
        logger.error(f"运行时错误: {e}")
        sys.exit(1)

def test_score_analysis():
    analyzer = ScoreAnalysis()
    scores = [0.85, 0.92, 0.78, 0.65, 0.59, 0.88, 0.73, 0.95, 0.82, 0.68]
    
    # 分析成绩
    analysis = analyzer.analyze_scores(scores)
    
    # 生成报告
    report = analyzer.generate_report(analysis)
    
    # 打印文本报告
    print(report["text_report"])
    
    # 可以将visualization保存为图片或在web界面显示
    # with open("score_distribution.png", "wb") as f:
    #     f.write(base64.b64decode(report["visualization"]))

if __name__ == "__main__":
    test_score_analysis()
