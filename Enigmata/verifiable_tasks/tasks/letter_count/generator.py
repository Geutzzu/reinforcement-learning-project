import random
import json
import os
from .template import PROMPT_TEMPLATE
from tqdm import tqdm

SYSTEM_WORDS = None

def load_system_dictionary():
    """Load words from system dictionary if available."""
    global SYSTEM_WORDS
    if SYSTEM_WORDS is not None:
        return SYSTEM_WORDS
    
    dict_paths = [
        "/usr/share/dict/words",           # macOS/Linux
        "/usr/share/dict/american-english", # Debian/Ubuntu
    ]
    
    for path in dict_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    SYSTEM_WORDS = [line.strip().lower() for line in f 
                                   if line.strip().isalpha() and len(line.strip()) >= 3]
                print(f"Loaded {len(SYSTEM_WORDS)} words from {path}")
                return SYSTEM_WORDS
            except:
                continue
    
    return None


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


def generate_pronounceable_word(min_len: int, max_len: int) -> str:
    """Generate a pronounceable word by alternating consonants and vowels."""
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    # Common consonant clusters for more natural words
    clusters = ["bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", 
                "pl", "pr", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", 
                "str", "sw", "th", "tr", "tw", "wh", "wr"]
    
    length = random.randint(min_len, max_len)
    word = ""
    use_vowel = random.choice([True, False])
    
    while len(word) < length:
        if use_vowel:
            # Sometimes use double vowels
            if random.random() < 0.2 and len(word) < length - 1:
                word += random.choice(vowels) * 2
            else:
                word += random.choice(vowels)
        else:
            # Sometimes use consonant clusters
            if random.random() < 0.3 and len(word) < length - 2:
                word += random.choice(clusters)
            else:
                word += random.choice(consonants)
        use_vowel = not use_vowel
    
    return word[:length]  # Trim to exact length


# Cache for generated words
GENERATED_MEDIUM_WORDS = None
GENERATED_HARD_WORDS = None


def get_generated_words(difficulty):
    """Get or generate synthetic words for medium/hard difficulties."""
    global GENERATED_MEDIUM_WORDS, GENERATED_HARD_WORDS
    
    if difficulty == 'medium':
        if GENERATED_MEDIUM_WORDS is None:
            # Generate 1000 words between 15-40 characters
            GENERATED_MEDIUM_WORDS = [generate_pronounceable_word(15, 40) for _ in range(1000)]
            print(f"Generated {len(GENERATED_MEDIUM_WORDS)} synthetic medium words (15-40 chars)")
        return GENERATED_MEDIUM_WORDS
    
    elif difficulty == 'hard':
        if GENERATED_HARD_WORDS is None:
            # Generate 500 words between 40-100 characters
            GENERATED_HARD_WORDS = [generate_pronounceable_word(40, 100) for _ in range(500)]
            print(f"Generated {len(GENERATED_HARD_WORDS)} synthetic hard words (40-100 chars)")
        return GENERATED_HARD_WORDS
    
    return []


def get_word_list(difficulty):
    """Get word list for difficulty, using system dict + generated words."""
    sys_words = load_system_dictionary()

    if not sys_words:
        print("No system dictionary found, using hardcoded lists")
        if difficulty == 'easy':
            return EASY_WORDS
        elif difficulty == 'medium':
            return MEDIUM_WORDS
        else:
            return HARD_WORDS
    
    if difficulty == 'easy':
        # Easy: dict words 3-7 characters
        filtered = [w for w in sys_words if 3 <= len(w) <= 7]
        return filtered if len(filtered) >= 100 else EASY_WORDS
    
    elif difficulty == 'medium':
        # Medium: dict words 7-15 chars + 30% generated words (15-40 chars)
        dict_words = [w for w in sys_words if 7 <= len(w) <= 15]
        generated = get_generated_words('medium')
        
        # Mix: 70% dict, 30% generated
        num_generated = int(len(dict_words) * 0.3 / 0.7)  # Calculate to get 30% ratio
        combined = dict_words + generated[:num_generated]
        random.shuffle(combined)
        return combined if len(combined) >= 100 else MEDIUM_WORDS
    
    else:  # hard
        # Hard: dict words > 15 chars + generated words (40-100 chars)
        dict_words = [w for w in sys_words if len(w) > 15]
        generated = get_generated_words('hard')
        
        # Combine all
        combined = dict_words + generated
        random.shuffle(combined)
        return combined if len(combined) >= 50 else HARD_WORDS


def count_letter(word: str, letter: str) -> int:
    """Count occurrences of letter in word."""
    return word.lower().count(letter.lower())


def generate_step_reasoning(word: str, letter: str) -> str:
    """Generate step-by-step reasoning for counting a letter."""
    lines = []
    count = 0
    
    for i, char in enumerate(word.lower()):
        if char == letter.lower():
            count += 1
            lines.append(f"{char} - YES ({count})")
        else:
            lines.append(f"{char} - no")
    
    return '\n'.join(lines)


def generate_exhaustive_reasoning(word: str, letter: str, correct_count: int) -> str:
    """Generate complete reasoning for the answer."""
    parts = []
    parts.append(f"Counting '{letter}' in \"{word}\":")
    parts.append("")
    parts.append(generate_step_reasoning(word, letter))
    parts.append("")
    parts.append(f"Total count: {correct_count}")
    
    return '\n'.join(parts)


def get_answer(word: str, letter: str, lang='en') -> str:
    """Generate the complete answer with reasoning."""
    correct_count = count_letter(word, letter)
    reasoning = generate_exhaustive_reasoning(word, letter, correct_count)
    
    return f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
{correct_count}
---end_answer---"""


def generate(count=100, difficulty='medium', language='en', split="train"):
    prompt_template = PROMPT_TEMPLATE
    word_list = get_word_list(difficulty)
    
    for i in tqdm(range(count)):
        word = random.choice(word_list)
        letter = random.choice(LETTERS)
        
        correct_count = count_letter(word, letter)
        
        attempts = 0
        while correct_count == 0 and attempts < 10:
            letter = random.choice(word)  # Pick a letter that exists in word
            correct_count = count_letter(word, letter)
            attempts += 1
        
        answer = get_answer(word, letter, lang=language)
        
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
                "word_length": len(word),
                "split": split,
                "type": "counting",
                "source_url": "auto-generated",
                "dataset_name": "letter_count",
                "difficulty_level": difficulty,
                "language": language,
            }),
        }
