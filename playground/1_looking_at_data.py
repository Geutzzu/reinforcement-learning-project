import sys

sys.path.insert(0, "/home/geo/rl")

from Enigmata.verifiable_tasks.tasks.maze.generator import generate
from Enigmata.verifiable_tasks.tasks.maze.verifier import verify

from main.enigmata import generate_puzzles, verify, AVAILABLE_TASKS
from utils.processing_tools import batch_tokenize

df = generate_puzzles("maze", count=10000, difficulty="hard")
df['tokens_len'] = batch_tokenize(df['prompt'] + df['answer'])
df.to_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")


## testing
import pandas as pd

df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
df = df.sample(5, random_state=42)
print(df['prompt'].iloc[0])
print(df['answer'].iloc[0])