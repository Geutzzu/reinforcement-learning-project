PROMPT_TEMPLATE = """Count how many times the letter '{letter}' appears in the word "{word}".

**STRICT OUTPUT FORMAT - DO NOT DEVIATE:**

Your response MUST contain ONLY these two blocks with NO text before, between, or after them.

---start_reasoning---
<go through each letter one by one>
---end_reasoning---
---start_answer---
<single number>
---end_answer---

CRITICAL:
- Go through each letter and mark YES or no
- Keep reasoning concise
- NO tokens outside the two blocks
- Answer block: ONLY the count as a single number
"""
