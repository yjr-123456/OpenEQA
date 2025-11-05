import os
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)
client = OpenAI(
    base_url="https://xiaoai.plus/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def compare_answers_with_api(agent_answer, ground_truth, question_stem="", question_type=""):

    if agent_answer is None or ground_truth is None:
        return False
    
    # 首先尝试简单的字符串匹配（快速判断明显相同的答案）
    agent_answer_str = str(agent_answer).strip()
    ground_truth_str = str(ground_truth).strip()
    
    if agent_answer_str.lower() == ground_truth_str.lower():
        return True
    
    # 如果简单匹配失败，调用API进行智能判断
    return call_api_for_answer_comparison(agent_answer_str, ground_truth_str, question_stem, question_type)

def call_api_for_answer_comparison(agent_answer, ground_truth, question_stem="", question_type=""):

    # 构建系统提示
    system_prompt = """You are an expert evaluator for question-answering tasks. Your job is to determine if an agent's answer equals to the ground truth answer.
        Return only "CORRECT" if the answers match or "INCORRECT" if they don't match.
        Do not provide explanations, just the verdict."""

    # 构建用户提示
    user_prompt = f"""Question Type: {question_type}
        Question: {question_stem}
        Ground Truth Answer: {ground_truth}
        Agent's Answer: {agent_answer}
        If the agent's answer is correct, return "CORRECT". If it is incorrect, return "INCORRECT".
        """

    try:

        response = client.chat.completions.create(
            model='gemini-2.0-flash-lite',  
            max_tokens=50,  
            temperature=0.3,  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # 解析结果
        if "CORRECT" == result:
            return True
        elif "INCORRECT" == result:
            return False
        else:
            # 如果API返回了意外的格式，回退到原始方法
            print(f"Warning: Unexpected API response: {result}. Falling back to string comparison.")
            return fallback_compare_answers(agent_answer, ground_truth)
            
    except Exception as e:
        print(f"Error calling API for answer comparison: {e}")
        # API调用失败时回退到原始方法
        return fallback_compare_answers(agent_answer, ground_truth)

def fallback_compare_answers(agent_answer, ground_truth):
    """
    API调用失败时的回退方法（原始的字符串比较逻辑）
    """
    if agent_answer is None or ground_truth is None:
        return False
    
    # 转换为字符串并标准化
    agent_answer = str(agent_answer).strip().lower()
    ground_truth = str(ground_truth).strip().lower()
    
    # 直接比较
    if agent_answer == ground_truth:
        return True
    
    # 处理选择题情况（A、B、C、D）
    if len(agent_answer) == 1 and len(ground_truth) == 1:
        return agent_answer == ground_truth
    
    # 处理数字答案
    try:
        agent_num = float(agent_answer)
        truth_num = float(ground_truth)
        return abs(agent_num - truth_num) < 0.01  # 允许小的浮点误差
    except ValueError:
        pass
    
    # 检查关键词包含关系
    if agent_answer in ground_truth or ground_truth in agent_answer:
        return True
    return False

def calculate_accuracy(correct, total):
    """计算准确率"""
    if total == 0:
        return 0.0
    return (correct / total) * 100
