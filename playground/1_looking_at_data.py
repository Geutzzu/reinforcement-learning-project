import sys
import pandas as pd 

sys.path.insert(0, "/home/geo/rl")


from main.enigmata import generate_puzzles, verify, AVAILABLE_TASKS
from utils.processing_tools import batch_tokenize

df = generate_puzzles("maze_2", count=22500, difficulty="hard")
df['tokens_len'] = batch_tokenize(df['prompt'] + df['answer'])

solvable = df[df['answer'] != "not exist"]
unsolvable = df[df['answer'] == "not exist"]
print(f"Solvable: {len(solvable)}, Unsolvable: {len(unsolvable)}")

n_unsolvable_target = int(len(solvable) * 0.20)  
unsolvable_sampled = unsolvable.sample(n=min(n_unsolvable_target, len(unsolvable)))
df_balanced = pd.concat([solvable, unsolvable_sampled]).sample(frac=1).reset_index(drop=True)
print(f"Balanced: {len(df_balanced)} total")

df_balanced.to_parquet("/Users/geo/facultate/rl/rl/data/maze_2.parquet")


## testing
import pandas as pd

df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
df = df.sample(5, random_state=42)
print(df['prompt'].iloc[0])
print(df['answer'].iloc[0]) 



