import random
import json
from .template import PROMPT_TEMPLATE
from tqdm import tqdm

# Word lists by difficulty
EASY_WORDS = [
    "apple", "banana", "orange", "grape", "lemon",
    "hello", "world", "python", "coding", "simple",
    "house", "mouse", "cat", "dog", "bird",
    "table", "chair", "book", "pencil", "paper",
]

MEDIUM_WORDS = [
    "strawberry", "blueberry", "raspberry", "watermelon", "pineapple",
    "computer", "keyboard", "programming", "algorithm", "function",
    "beautiful", "wonderful", "fantastic", "excellent", "brilliant",
    "elephant", "giraffe", "butterfly", "caterpillar", "grasshopper",
]

HARD_WORDS = [
    "supercalifragilisticexpialidocious", "antidisestablishmentarianism",
    "pneumonoultramicroscopicsilicovolcanoconiosis", "floccinaucinihilipilification",
    "hippopotomonstrosesquippedaliophobia", "pseudopseudohypoparathyroidism",
    "incomprehensibility", "internationalization", "counterrevolutionary",
    "electroencephalograph", "otorhinolaryngology", "psychophysicotherapeutics",
    "uncharacteristically", "disproportionateness", "indistinguishability",
]

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def count_letter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())


def generate(count=100, difficulty='medium', language='en', split="train"):
    prompt_template = PROMPT_TEMPLATE
    
    if difficulty == 'easy':
        word_list = EASY_WORDS
    elif difficulty == 'medium':
        word_list = MEDIUM_WORDS
    else:
        word_list = HARD_WORDS
    
    for i in tqdm(range(count)):
        word = random.choice(word_list)
        letter = random.choice(LETTERS)
        
        correct_count = count_letter(word, letter)
        
        # Ensure we have some variety (not always 0)
        attempts = 0
        while correct_count == 0 and attempts < 10:
            letter = random.choice(word)  # Pick a letter that exists in word
            correct_count = count_letter(word, letter)
            attempts += 1
        
        answer = str(correct_count)
        
        yield {
            "prompt": prompt_template.format(word=word, letter=letter),
            "answer": answer,
            "task_name": "letter_count",
            "ability": "reasoning",
            "language": language,
            "meta": json.dumps({
                "id": f"letter_count_{difficulty}_{i}",
                "word": word,
                "letter": letter,
                "count": correct_count,
                "answer": answer,
                "rationale": "",
                "split": split,
                "type": "counting",
                "source_url": "auto-generated",
                "dataset_name": "letter_count",
                "difficulty_level": difficulty,
                "language": language,
            }),
        }
