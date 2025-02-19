import unittest
import sys
import os
import logging
from datetime import datetime

# 配置日志
def setup_logging():
    log_filename = f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 创建 FileHandler 时指定编码
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)  # 使用 stdout 而不是默认的 stderr
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class DetailedTestResult(unittest.TestResult):
    def startTest(self, test):
        self.startTime = datetime.now()
        super().startTest(test)
        logger.info(f"\n=== 开始测试: {test._testMethodName} ===")
        logger.info(f"测试描述: {test.shortDescription() or '无描述'}")

    def addSuccess(self, test):
        elapsed = (datetime.now() - self.startTime).total_seconds()
        super().addSuccess(test)
        # 使用 PASS 替代 ✓ 符号
        logger.info(f"测试通过 [PASS] (耗时: {elapsed:.3f}秒)")

    def addError(self, test, err):
        elapsed = (datetime.now() - self.startTime).total_seconds()
        super().addError(test, err)
        # 使用 ERROR 替代 ✗ 符号
        logger.error(f"测试错误 [ERROR] (耗时: {elapsed:.3f}秒)")
        logger.error(f"错误信息: {err[1]}")

    def addFailure(self, test, err):
        elapsed = (datetime.now() - self.startTime).total_seconds()
        super().addFailure(test, err)
        # 使用 FAIL 替代 ✗ 符号
        logger.error(f"测试失败 [FAIL] (耗时: {elapsed:.3f}秒)")
        logger.error(f"失败信息: {err[1]}")

def run_tests():
    """运行所有测试并返回结果"""
    global logger
    logger = setup_logging()
    
    # 设置测试环境
    os.environ['TESTING'] = 'True'
    logger.info("=== 开始测试套件 ===")
    
    # 加载所有测试
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)

    # 运行测试
    runner = unittest.TextTestRunner(
        resultclass=DetailedTestResult,
        verbosity=2
    )
    result = runner.run(suite)
    
    # 输出测试统计
    logger.info("\n=== 测试套件完成 ===")
    logger.info(f"运行测试总数: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 