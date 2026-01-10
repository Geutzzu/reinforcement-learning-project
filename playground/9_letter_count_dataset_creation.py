import sys
import pandas as pd

sys.path.insert(0, "/Users/geo/facultate/rl/rl")

from main.enigmata import generate_puzzles
from utils.processing_tools import batch_tokenize
import Enigmata.verifiable_tasks.tasks.letter_count.generator

df = generate_puzzles("letter_count", count=10000, difficulty="very_hard")

df['tokens_len'] = batch_tokenize(df.apply(lambda x: x['prompt'] + "\n\n" + x['answer'], axis=1))

df.to_parquet("/Users/geo/facultate/rl/rl/data/rl_letter_count_dataset_very_hard.parquet", index=False)