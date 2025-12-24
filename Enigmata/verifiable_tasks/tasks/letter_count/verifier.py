"""
Letter counting verifier with partial rewards.

Reward breakdown:
- 0.2: Shows step-by-step reasoning (lists letters or uses dashes/commas)
- 0.1: Mentions the target letter in reasoning
- 0.2: Has a code block with a number
- 0.5: Correct answer (or 0.3 if off by 1)

Total: 1.0 for correct answer with reasoning
"""
import re
import json


def extract_last_code_block(text: str):
    code_blocks = re.findall(r'```.*?\n(.*?)```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    return code_blocks[-1].strip() if code_blocks else None


def verify(pred, answer, meta):
    score = 0.0
    
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    correct_count = meta["count"]
    target_letter = meta["letter"].lower()
    word = meta["word"].lower()
    
    # Reward 1: Shows reasoning (lists letters with separators)
    # Looking for patterns like "s-t-r-a-w" or "s, t, r, a, w" or letter enumeration
    reasoning_patterns = [
        r'[a-z]\s*[-,]\s*[a-z]',  # Letters separated by dash or comma
        r'\d+[.):]\s*[a-z]',  # Numbered list like "1. s"
        r"letter",  # Mentions "letter"
        r"count",  # Mentions counting
    ]
    has_reasoning = any(re.search(p, pred.lower()) for p in reasoning_patterns)
    if has_reasoning:
        score += 0.2
    
    # Reward 2: Mentions the target letter explicitly in reasoning
    letter_mention_patterns = [
        f"'{target_letter}'",
        f'"{target_letter}"',
        f"letter {target_letter}",
        f"'{target_letter}' appears",
        f"the {target_letter}",
    ]
    if any(p in pred.lower() for p in letter_mention_patterns):
        score += 0.1
    
    # Reward 3: Has code block with a number
    code_content = extract_last_code_block(pred)
    if code_content and re.search(r'\d+', code_content):
        score += 0.2
    
    # Reward 4: Correct answer
    # Try to extract the final answer from code block first, then from text
    final_answer = None
    
    if code_content:
        numbers = re.findall(r'\b(\d+)\b', code_content)
        if numbers:
            final_answer = int(numbers[-1])
    
    if final_answer is None:
        # Fall back to last number in the text
        numbers = re.findall(r'\b(\d+)\b', pred)
        if numbers:
            final_answer = int(numbers[-1])
    
    if final_answer is not None:
        if final_answer == correct_count:
            score = 1.0  # Perfect score for correct answer
        elif abs(final_answer - correct_count) == 1:
            score = max(score, 0.5)  # Partial credit for off-by-one
        elif abs(final_answer - correct_count) <= 2:
            score = max(score, 0.4)  # Small partial credit for close
    
    return score
