import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.font_manager as fm

class ScoreAnalysis:
    def __init__(self):
        self.stats = {}
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    def _get_distribution(self, scores: List[float]) -> Dict[str, int]:
        """
        计算成绩分布
        """
        ranges = {
            "优秀(90-100)": 0,
            "良好(80-89)": 0,
            "中等(70-79)": 0,
            "及格(60-69)": 0,
            "不及格(0-59)": 0
        }
        
        for score in scores:
            score = score * 100  # 转换为百分制
            if score >= 90:
                ranges["优秀(90-100)"] += 1
            elif score >= 80:
                ranges["良好(80-89)"] += 1
            elif score >= 70:
                ranges["中等(70-79)"] += 1
            elif score >= 60:
                ranges["及格(60-69)"] += 1
            else:
                ranges["不及格(0-59)"] += 1
                
        return ranges

    def analyze_scores(self, scores: List[float]) -> Dict[str, Any]:
        """
        分析成绩分布
        """
        if not scores:
            raise ValueError("成绩列表不能为空")
            
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
            "distribution": self._get_distribution(scores)
        }
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果字典
            
        Returns:
            Dict包含报告文本和可视化图表
        """
        # 生成文本报告
        report_text = (
            f"成绩分析报告:\n"
            f"1. 平均分: {analysis_results['mean']:.2f}\n"
            f"2. 中位数: {analysis_results['median']:.2f}\n"
            f"3. 标准差: {analysis_results['std']:.2f}\n"
            f"4. 最高分: {analysis_results['max']:.2f}\n"
            f"5. 最低分: {analysis_results['min']:.2f}\n\n"
            f"成绩分布:\n"
        )
        
        for range_name, count in analysis_results['distribution'].items():
            report_text += f"{range_name}: {count}人\n"
            
        # 生成可视化图表
        plt.figure(figsize=(10, 6))
        plt.bar(
            analysis_results['distribution'].keys(),
            analysis_results['distribution'].values()
        )
        plt.title("成绩分布图")
        plt.xlabel("分数段")
        plt.ylabel("人数")
        plt.xticks(rotation=45)
        
        # 将图表转换为base64字符串
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "text_report": report_text,
            "visualization": image_base64,
            "statistics": analysis_results
        } 