import json
import re
from numpy import sort_complex
import pandas as pd
from Enigmata.verifiable_tasks.tasks.maze_2.verifier import verify, verify_format, verify_answer

# =============================================================================
# TEST EXAMPLES
# =============================================================================

# UNSOLVABLE MAZE (for "not exist" tests)
META_UNSOLVABLE = {
    "question": [
        ['S', '.', '.', '.', '.'],
        ['.', 'B', '.', '.', '.'],
        ['B', '.', '.', 'B', 'B'],
        ['B', '.', 'B', 'B', '.'],
        ['.', 'B', '.', 'B', 'E']
    ],
    "height": 5,
    "width": 5
}

# SOLVABLE MAZE (for path tests)
# Valid path: (1,1)->(1,2)->(1,3)->(2,3)->(3,3)->(3,2)->(4,2)->(5,2)->(5,3)->(5,4)->(5,5)
META_SOLVABLE = {
    "question": [
        ['S', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', 'E']
    ],
    "height": 5,
    "width": 5
}

# Example 1: CORRECT "not exist" answer
PRED_CORRECT_NOT_EXIST = """---start_reasoning---
After careful analysis of all possible paths, I found that the maze is blocked.
---end_reasoning---
---start_answer---
not exist
---end_answer---"""

# Example 2: CORRECT path answer
PRED_CORRECT_PATH = """---start_reasoning---
Starting from (1,1), I'll navigate to (5,5).
Going right then down should work.
---end_reasoning---
---start_answer---
(1,1)->(1,2)->(1,3)->(1,4)->(1,5)->(2,5)->(3,5)->(4,5)->(5,5)
---end_answer---"""

# Example 3: MAX TOKENS exploit (missing ---end_answer---)
PRED_MAX_TOKENS_EXPLOIT = """---start_reasoning---
Let me analyze this maze step by step.
---end_reasoning---
---start_answer---
(1,1)->(1,2)->(1,3) and I need to continue but"""

# Example 4: SHORT PATH exploit (lazy 2-step answer on SOLVABLE maze)
PRED_SHORT_PATH_EXPLOIT = """---start_reasoning---
From (1,1), I can move to (1,2).
---end_reasoning---
---start_answer---
(1,1)->(1,2)
---end_answer---"""

# Example 5: WRONG ANSWER - says path when answer is "not exist"
PRED_PATH_WHEN_NOT_EXIST = """---start_reasoning---
I think there's a path.
---end_reasoning---
---start_answer---
(1,1)->(1,2)->(1,3)->(1,4)->(1,5)->(2,5)->(3,5)->(4,5)->(5,5)
---end_answer---"""

# Example 6: PATH HITS OBSTACLE (wrong path on solvable maze)
PRED_HITS_OBSTACLE = """---start_reasoning---
Going down from start.
---end_reasoning---
---start_answer---
(1,1)->(2,1)->(3,1)->(4,1)->(5,1)->(5,2)->(5,3)->(5,4)->(5,5)
---end_answer---"""

# Example 7: ALMOST CORRECT - stops 1 step before goal
PRED_ALMOST_CORRECT = """---start_reasoning---
Navigating through the maze.
---end_reasoning---
---start_answer---
(1,1)->(1,2)->(1,3)->(1,4)->(1,5)->(2,5)->(3,5)->(4,5)
---end_answer---"""

ANSWER_NOT_EXIST = "not exist"
ANSWER_PATH = "(1,1)->(1,2)->(1,3)->(1,4)->(1,5)->(2,5)->(3,5)->(4,5)->(5,5)"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REWARD FUNCTION TESTS")
    print("=" * 60)
    
    test_cases = [
        # (name, prediction, expected_answer, maze_meta)
        ("CORRECT: not exist", PRED_CORRECT_NOT_EXIST, ANSWER_NOT_EXIST, META_UNSOLVABLE),
        ("CORRECT: valid path", PRED_CORRECT_PATH, ANSWER_PATH, META_SOLVABLE),
        ("EXPLOIT: max tokens (no end marker)", PRED_MAX_TOKENS_EXPLOIT, ANSWER_PATH, META_SOLVABLE),
        ("EXPLOIT: short lazy path", PRED_SHORT_PATH_EXPLOIT, ANSWER_PATH, META_SOLVABLE),
        ("WRONG: path when not exist", PRED_PATH_WHEN_NOT_EXIST, ANSWER_NOT_EXIST, META_UNSOLVABLE),
        ("WRONG: almost correct (1 step short)", PRED_ALMOST_CORRECT, ANSWER_PATH, META_SOLVABLE),
    ]
    
    for name, pred, answer, meta in test_cases:
        print(f"\n--- {name} ---")
        print(f"verify_format:       {verify_format(pred, answer, meta):.2f}")
        print(f"verify_answer:       {verify_answer(pred, answer, meta):.2f}")
        print(f"verify (combined):   {verify(pred, answer, meta):.2f}")
    
    print("\n" + "=" * 60)
    print("EXPECTED SCORES:")
    print("- CORRECT not exist:     ~2.50 (format + answer)")
    print("- CORRECT valid path:    ~2.50+ (format + answer + completion)")
    print("- EXPLOIT max tokens:    ~-1.00 (format penalty)")
    print("- EXPLOIT short path:    ~0.30 (lazy path penalty)")
    print("- WRONG path when none:  ~0.70 (wrong answer type)")
    print("- WRONG almost correct:  ~1.50+ (good progress, no completion)")
    print("=" * 60)
