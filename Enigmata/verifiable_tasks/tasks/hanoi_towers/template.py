PROMPT_TEMPLATE = """Solve the Tower of Hanoi puzzle.

Current State:
{question}

Rules:
1. Only one disk can be moved at a time.
2. A disk can only be placed on top of a larger disk or on an empty peg.
3. Only the top disk of any peg can be moved.
4. Goal: Move ALL disks from Peg A to Peg C.

**STRICT OUTPUT FORMAT - DO NOT DEVIATE:**

Your response MUST contain ONLY these two blocks with NO text before, between, or after them.

---start_reasoning---
<brief step-by-step reasoning>
---end_reasoning---
---start_answer---
A->C, A->B, C->B, ...
---end_answer---

CRITICAL:
- Keep reasoning SHORT and to the point
- NO tokens outside the two blocks
- Answer block: ONLY comma-separated moves in format X->Y where X,Y are A, B, or C
- Each move transfers the top disk from peg X to peg Y
"""
