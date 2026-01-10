PROMPT_TEMPLATE_OLD_USED_ON_SFT = """Given a 5x5 maze map, as shown below:

{question}

Where:
S represents the start point (located in the top-left corner at coordinates (1, 1))
E represents the end point (located in the bottom-right corner at coordinates (5, 5))
B represents an obstacle (impassable)
. represents open space (passable)

Rules:
1. You can only move up, down, left, or right, not diagonally.
2. You cannot pass through obstacles (B).
3. You can move freely on open spaces (.).
4. The goal is to find a path from the start point (S) to the end point (E).

Find a valid path from start to end.

**Response format (you MUST follow this exactly):**

---start_reasoning---
<your step-by-step reasoning here>
---end_reasoning---

---start_answer---
(1,1)->(row,col)->...->(5,5)
---end_answer---

Rules:
- Your reasoning goes between the reasoning markers (free form text)
- Answer block must contain ONLY the path: coordinates separated by ->
- If no path exists, answer block must contain exactly: 'not exist'
"""






PROMPT_TEMPLATE = """Given a 5x5 maze map, as shown below:

{question}

Where:
S represents the start point (located in the top-left corner at coordinates (1, 1))
E represents the end point (located in the bottom-right corner at coordinates (5, 5))
B represents an obstacle (impassable)
. represents open space (passable)

Rules:
1. You can only move up, down, left, or right, not diagonally.
2. You cannot pass through obstacles (B).
3. You can move freely on open spaces (.).
4. The goal is to find a path from the start point (S) to the end point (E).

Find a valid path from start to end.

**STRICT OUTPUT FORMAT - DO NOT DEVIATE:**

Your response MUST contain ONLY these two blocks with NO text before, between, or after them.

If path EXISTS:
---start_reasoning---
<brief reasoning>
---end_reasoning---
---start_answer---
(1,1)->(r,c)->...->(5,5)
---end_answer---

If NO path exists:
---start_reasoning---
<brief explanation why no path>
---end_reasoning---
---start_answer---
not exist
---end_answer---

CRITICAL:
- Keep reasoning SHORT and to the point
- NO tokens outside the two blocks
- Answer block: ONLY the path OR "not exist"
"""
