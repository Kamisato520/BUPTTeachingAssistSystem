import unittest
import numpy as np
from plugins.score_analysis import ScoreAnalysis
import logging

logger = logging.getLogger(__name__)

class TestScoreAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = ScoreAnalysis()
        self.test_scores = [0.85, 0.92, 0.78, 0.65, 0.59, 0.88, 0.73, 0.95, 0.82, 0.68]
        logger.info(f"测试数据集: {self.test_scores}")

    def test_analyze_scores(self):
        """测试成绩分析功能"""
        logger.info("\n测试成绩分析功能")
        logger.info(f"输入成绩: {self.test_scores}")
        
        analysis = self.analyzer.analyze_scores(self.test_scores)
        
        logger.info("分析结果:")
        logger.info(f"平均分: {analysis['mean']:.2f}")
        logger.info(f"中位数: {analysis['median']:.2f}")
        logger.info(f"标准差: {analysis['std']:.2f}")
        logger.info(f"最高分: {analysis['max']:.2f}")
        logger.info(f"最低分: {analysis['min']:.2f}")
        logger.info(f"分布情况: {analysis['distribution']}")
        
        self.assertAlmostEqual(analysis['mean'], np.mean(self.test_scores))
        self.assertAlmostEqual(analysis['max'], max(self.test_scores))
        self.assertAlmostEqual(analysis['min'], min(self.test_scores))

    def test_generate_report(self):
        """测试报告生成功能"""
        logger.info("\n测试报告生成功能")
        
        analysis = self.analyzer.analyze_scores(self.test_scores)
        report = self.analyzer.generate_report(analysis)
        
        logger.info("生成的报告:")
        logger.info(report['text_report'])
        logger.info(f"可视化图表大小: {len(report['visualization'])} bytes")
        
        self.assertIn('text_report', report)
        self.assertIn('visualization', report)
        self.assertIn('statistics', report)

if __name__ == '__main__':
    unittest.main() 