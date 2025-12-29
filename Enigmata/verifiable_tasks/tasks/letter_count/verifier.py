import re
import json


def extract_reasoning(text: str) -> str:
    pattern = r'---start_reasoning---(.*?)---end_reasoning---'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer(text: str) -> str:
    pattern = r'---start_answer---(.*?)---end_answer---'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def verify_format(pred, answer, meta):
    reasoning = extract_reasoning(pred)
    answer_block = extract_answer(pred)
    
    if reasoning is not None and answer_block is not None:
        return 1.0
    elif reasoning is not None or answer_block is not None:
        return 0.5
    return 0.0


def verify_answer(pred, answer, meta):
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    correct_count = meta["count"]
    
    answer_block = extract_answer(pred)
    if answer_block is None:
        return -0.5
    
    numbers = re.findall(r'\b(\d+)\b', answer_block)
    if not numbers:
        return -0.3
    
    final_answer = int(numbers[0])
    
    if final_answer == correct_count:
        return 1.0
    elif abs(final_answer - correct_count) == 1:
        return 0.3
    elif abs(final_answer - correct_count) <= 2:
        return 0.1
    else:
        return -0.3


def verify(pred, answer, meta):
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    format_score = verify_format(pred, answer, meta) * 0.3
    
    answer_score = verify_answer(pred, answer, meta)
    if answer_score < 0:
        answer_contribution = answer_score * 0.5
    else:
        answer_contribution = answer_score * 0.7
    
    total = format_score + answer_contribution
    return max(0.0, min(1.0, total)) 
