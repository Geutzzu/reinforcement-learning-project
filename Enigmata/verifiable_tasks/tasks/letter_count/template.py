PROMPT_TEMPLATE = """Count how many times the letter '{letter}' appears in the word "{word}".

Think step by step:
1. Write out each letter of the word
2. Mark which ones match '{letter}'
3. Count the total

**Response format (you MUST follow this exactly):**

---start_reasoning---
<your step-by-step reasoning here>
---end_reasoning---

---start_answer---
<your final count as a single number>
---end_answer---

Example for counting 'a' in "banana":
---start_reasoning---
Let me go through each letter:
b - not 'a'
a - yes, count = 1
n - not 'a'
a - yes, count = 2
n - not 'a'
a - yes, count = 3
Total: 3
---end_reasoning---

---start_answer---
3
---end_answer---

Now count the '{letter}'s in "{word}":
"""
