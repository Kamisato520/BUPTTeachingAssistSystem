import uuid
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_chroma import Chroma

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# ========== 数据类定义 ==========

@dataclass
class Question:
    id: str
    content: str
    question_type: str
    answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuestionEvaluation:
    question_id: str
    score: float
    is_new: bool

# ========== 示例性占位 ==========

# 假设已经有一个 Chroma 知识库对象 vectorstore
# 您需要提前构建并持久化，比如使用:
# vectorstore = Chroma(persist_directory="path_to_chroma_db")
# 这里直接假设已经有 vectorstore
keys = "sk-WKGLTTbN0d583D64555bT3BlBkFJ50E2848daCF9409AAe18"
# 创建Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=keys, openai_api_base="https://cfcus02.opapi.win/v1")

# 创建Vectorstore时提供embedding_function
vectorstore = Chroma(
    persist_directory="../knowledge_base_demo.db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# 定义LLM
llm_A = OpenAI(
    temperature=0, 
    openai_api_key=keys,
    openai_api_base="https://cfcus02.opapi.win/v1"
)  # 假设使用OpenAI作为LLM-A
llm_B = OpenAI(
    temperature=0, 
    openai_api_key=keys,
    openai_api_base="https://cfcus02.opapi.win/v1"
  )  # 假设使用OpenAI作为LLM-B, 更高能力模型，比如GPT-4

# ========== LangChain相关 Chain 定义 ==========

# 1. 大语言模型A理解教师给出的提示词，拆分生成任务描述
task_understanding_prompt = PromptTemplate(
    input_variables=["teacher_prompt"],
    template=(
        "你是一个助教，请根据以下提示理解和拆分任务，用更适合生成题目的描述语言进行表述。\n"
        "教师提示：{teacher_prompt}\n"
        "请输出新的任务描述："
    )
)
task_chain = LLMChain(llm=llm_A, prompt=task_understanding_prompt)

# 2. 使用LLM-A + RAG生成新题目


# 定义期望的输出结构，比如需要一个JSON数组，每个元素有question_type、content、answer、options字段
question_schema = ResponseSchema(
    name="questions",
    description="A JSON array of question objects, each with question_type, content, answer, and options (for multiple_choice)."
)

parser = StructuredOutputParser.from_response_schemas([question_schema])
format_instructions = parser.get_format_instructions()

generate_question_prompt = PromptTemplate(
    input_variables=["task_description", "retrieved_knowledge"],
    template=(
    "下面是为出题的任务描述：{task_description}\n\n"
    "下面是检索到的相关知识内容：\n{retrieved_knowledge}\n\n"
    "请根据上述描述和知识点生成若干新题目，以 JSON 数组格式返回，严格按照以下格式：\n"
    "{{\n"
    "  \"questions\": [\n"
    "    {{\n"
    "      \"question_type\": \"multiple_choice\",\n"
    "      \"content\": \"...\",\n"
    "      \"answer\": \"...\",\n"
    "      \"options\": [\"A\", \"B\", \"C\", \"D\"]\n"
    "    }},\n"
    "    {{\n"
    "      \"question_type\": \"short_answer\",\n"
    "      \"content\": \"...\",\n"
    "      \"answer\": \"...\"\n"
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "请勿输出多余的说明，只输出上述格式的JSON。"
)
)

def generate_new_questions_from_llm(task_description: str, retriever: BaseRetriever) -> List[Question]:
    docs = retriever.get_relevant_documents(task_description)
    retrieved_texts = "\n".join([d.page_content for d in docs])
    
    gen_chain = LLMChain(llm=llm_A, prompt=generate_question_prompt)
    output = gen_chain.run(
        task_description=task_description, 
        retrieved_knowledge=retrieved_texts,
        question_type="",  # 添加缺少的参数
        questions=""       # 添加缺少的参数
    )
    
    try:
        parsed = parser.parse(output)  # 使用StructuredOutputParser解析
        questions_data = parsed["questions"]  # 根据上面定义的schema获取JSON数组
    except Exception as e:
        print("解析LLM输出时出现异常:", e)
        print("LLM Output:", output)
        questions_data = []

    new_questions = []
    for qd in questions_data:
        q_id = str(uuid.uuid4())
        question_type = qd.get("question_type", "short_answer")
        content = qd.get("content", "")
        answer = qd.get("answer", "")
        options = qd.get("options", [])

        new_questions.append(
            Question(
                id=q_id,
                content=content,
                question_type=question_type,
                answer=answer,
                metadata={"options": options}
            )
        )

    return new_questions

# 3. 检验旧题目
def check_old_questions(old_questions: List[Question], retriever: BaseRetriever) -> List[Question]:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    check_old_prompt = PromptTemplate(
        input_variables=["question_content", "retrieved_knowledge"],
        template=(
            "以下是一个旧题目：\n{question_content}\n\n"
            "下面是从知识库中检索到的最新相关知识内容：\n{retrieved_knowledge}\n\n"
            "请根据上述最新知识内容判断该题目是否已经过时或不再适用于当前教学需求。\n"
            "如果过时，请回答“过时”两个字，否则回答“不过时”两个字。"
        )
    )

    llm_A_for_check = llm_A  
    check_old_chain = LLMChain(llm=llm_A_for_check, prompt=check_old_prompt)

    updated_questions = []
    for old_q in old_questions:
        docs = retriever.get_relevant_documents(old_q.content)
        retrieved_texts = "\n".join([d.page_content for d in docs])

        response = check_old_chain.run(
            question_content=old_q.content,
            retrieved_knowledge=retrieved_texts
        ).strip()

        if response == "不过时":
            updated_questions.append(old_q)

    return updated_questions

# 4. 使用LLM-B对题目进行评分
scoring_prompt = PromptTemplate(
    input_variables=["question_content", "question_answer", "extra_dimension", "question_status"],
    template=(
        "请对以下题目进行质量评估并给出0-100的整数分数。\n\n"
        "题目（{question_status}）：{question_content}\n答案：{question_answer}\n"
        "维度包含：忠诚性、答案相关性、上下文相关性{extra_dimension}\n"
        "请仅输出一个0-100的数字评分。"
    )
)

def evaluate_questions_with_llm_B(questions: List[Question], is_new: bool) -> List[QuestionEvaluation]:
    evaluations = []
    extra_dimension = "、与题库原有题目相似度" if is_new else ""
    question_status = "新题" if is_new else "旧题"
    
    for q in questions:
        chain = LLMChain(llm=llm_B, prompt=scoring_prompt)
        score_str = chain.run(
            question_content=q.content,
            question_answer=q.answer,
            extra_dimension=extra_dimension,
            question_status=question_status
        )
        try:
            score = float(score_str.strip())
        except:
            score = random.uniform(0,100)
        
        evaluations.append(QuestionEvaluation(question_id=q.id, score=score, is_new=is_new))
    return evaluations

# ========== 主流程类 ==========

class QuestionPipeline:
    def __init__(self, 
                 retriever: BaseRetriever,
                 high_threshold: float = 80.0,
                 low_threshold: float = 40.0):
        self.retriever = retriever
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.question_bank = []
        self.discarded_bank = []

    def run_pipeline(self, teacher_prompt: str, old_questions: List[Question]) -> Dict[str, Any]:
        # 1. 使用LLM-A理解任务
        task_description = task_chain.run(teacher_prompt=teacher_prompt)
        
        # 2. 生成新题
        new_questions = generate_new_questions_from_llm(task_description, self.retriever)
        
        # 3. 检验旧题
        checked_old_questions = check_old_questions(old_questions, self.retriever)
        
        # 4. 使用LLM-B对新题、旧题打分
        new_evals = evaluate_questions_with_llm_B(new_questions, is_new=True)
        old_evals = evaluate_questions_with_llm_B(checked_old_questions, is_new=False)

        # 5. 三支决策
        self.three_way_decision(new_evals + old_evals, new_questions + checked_old_questions)
        
        # 6. 优化阈值（占位，可根据需要实现实际的优化算法）
        self.optimize_thresholds()

        # 在返回的字典中增加 "new_questions": new_questions
        return {
            "question_bank": self.question_bank,
            "discarded_bank": self.discarded_bank,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "new_questions": new_questions
        }

    def three_way_decision(self, evaluations: List[QuestionEvaluation], questions: List[Question]) -> None:
        q_map = {q.id: q for q in questions}
        for ev in evaluations:
            q = q_map[ev.question_id]
            if ev.score >= self.high_threshold:
                self.question_bank.append(q)
            elif ev.score < self.low_threshold:
                self.discarded_bank.append(q)
            else:
                # 中分数域题，需教师判断（此处简化逻辑）
                if "示例" in q.content:
                    self.discarded_bank.append(q)
                else:
                    self.question_bank.append(q)

    def optimize_thresholds(self):
        self.high_threshold += random.uniform(-1,1)
        self.low_threshold += random.uniform(-1,1)


# ========== 使用示例 ==========

if __name__ == "__main__":
    pipeline = QuestionPipeline(retriever=retriever, high_threshold=85.0, low_threshold=50.0)

    old_questions_example = [
        Question(
            id="old_q_1",
            content="旧题目示例1：请解释价格弹性的定义。",
            question_type="short_answer",
            answer="价格弹性是指需求量对价格变化的敏感程度。",
        ),
        Question(
            id="old_q_2",
            content="旧题目示例2：过时的计算题。",
            question_type="multiple_choice",
            answer="正确选项"
        )
    ]

    teacher_prompt = "生成两个关于经济学供需关系的考题，包括1个选择题和1个简答题，并给出标准答案。"
    result = pipeline.run_pipeline(teacher_prompt, old_questions_example)

    print("题库中题目数:", len(result["question_bank"]))
    print("废弃题库中题目数:", len(result["discarded_bank"]))
    print("当前阈值设置:", result["high_threshold"], result["low_threshold"])

    # 输出新生成的题目内容
    print("新生成的题目:")
    for q in result["new_questions"]:
        print(f"ID: {q.id}")
        print(f"题型: {q.question_type}")
        print(f"题干: {q.content}")
        print(f"答案: {q.answer}")
        print(f"选项: {q.metadata.get('options', [])}")
        print("-" * 40)