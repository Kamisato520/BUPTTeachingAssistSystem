from sanic import Sanic, response
from sanic.request import Request
from typing import List, Dict
import uuid
import json

# 引入之前定义的类和函数
from optimized_question_system import Question, QuestionPipeline, retriever

from sanic_ext import Extend

# 初始化 Sanic 应用
app = Sanic("QuestionGenerationAPI")
Extend(app)  # 启用扩展，包括 CORS 支持


@app.middleware("response")
async def add_cors_headers(request, response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"


# 初始化 QuestionPipeline 实例
pipeline = QuestionPipeline(retriever=retriever, high_threshold=85.0, low_threshold=50.0)


# 定义 API 路由
@app.post("/generate_questions")
async def generate_questions(request: Request):
    """
    接收教师提示词和旧题目列表，调用 pipeline.run_pipeline 生成题目。
    """
    try:
        # 获取请求中的 JSON 数据
        request_data = request.json
        teacher_prompt = request_data.get("teacher_prompt", "")
        old_questions_data = request_data.get("old_questions", [])

        # 将旧题目数据转换为 Question 对象
        old_questions = [
            Question(
                id=q.get("id", str(uuid.uuid4())),
                content=q.get("content", ""),
                question_type=q.get("question_type", "short_answer"),
                answer=q.get("answer", ""),
                metadata=q.get("metadata", {})
            )
            for q in old_questions_data
        ]

        # 调用 pipeline 生成新题目
        result = pipeline.run_pipeline(teacher_prompt, old_questions)

        # 构造返回数据
        response_data = {
            "question_bank": [
                {
                    "id": q.id,
                    "content": q.content,
                    "question_type": q.question_type,
                    "answer": q.answer,
                    "metadata": q.metadata,
                }
                for q in result["question_bank"]
            ],
            "discarded_bank": [
                {
                    "id": q.id,
                    "content": q.content,
                    "question_type": q.question_type,
                    "answer": q.answer,
                    "metadata": q.metadata,
                }
                for q in result["discarded_bank"]
            ],
            "new_questions": [
                {
                    "id": q.id,
                    "content": q.content,
                    "question_type": q.question_type,
                    "answer": q.answer,
                    "metadata": q.metadata,
                }
                for q in result["new_questions"]
            ],
            "high_threshold": result["high_threshold"],
            "low_threshold": result["low_threshold"],
        }

        return response.json(response_data, status=200)

    except Exception as e:
        # 返回错误信息
        return response.json({"error": str(e)}, status=500)


# 启动 Sanic 服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
