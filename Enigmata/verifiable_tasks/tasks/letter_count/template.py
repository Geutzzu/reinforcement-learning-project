PROMPT_TEMPLATE = """Count how many times the letter '{letter}' appears in the word "{word}".

Think step by step:
1. Write out each letter of the word
2. Mark which ones match '{letter}'
3. Count the total

**Response format:**
Put your final answer as a single number inside a code block:
```
<number>
```

For example, if the answer is 5:
```
5
```

Now count the '{letter}'s in "{word}":
"""
