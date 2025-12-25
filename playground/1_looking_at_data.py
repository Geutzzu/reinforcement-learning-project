import sys
import pandas as pd 

sys.path.insert(0, "/home/geo/rl")


from main.enigmata import generate_puzzles, verify, AVAILABLE_TASKS
from utils.processing_tools import batch_tokenize

df = generate_puzzles("maze_2", count=22500, difficulty="hard")
df['tokens_len'] = batch_tokenize(df['answer'])

solvable = df[df['answer'].str.contains("not exist") == False]
unsolvable = df[df['answer'].str.contains("not exist")]
print(f"Solvable: {len(solvable)}, Unsolvable: {len(unsolvable)}")

n_unsolvable_target = int(len(solvable) * 0.25)
unsolvable_sampled = unsolvable.sample(n=min(n_unsolvable_target, len(unsolvable)))
df_balanced = pd.concat([solvable, unsolvable_sampled]).sample(frac=1).reset_index(drop=True)
print(f"Balanced: {len(df_balanced)} total")

df_balanced.to_parquet("/Users/geo/facultate/rl/rl/data/maze_2_reasoning.parquet")


## testing
import pandas as pd

df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
df = df.sample(5, random_state=42)
print(df['prompt'].iloc[0])
print(df['answer'].iloc[0]) 





 
#### pattern if no reasoning is wanted (for experiments)
# ---start_reasoning---
# Solving the maze using pathfinding.
# ---end_reasoning---

# ---start_answer---
# (1,1)->(2,1)->(3,1)->(3,2)->(3,3)->(4,3)->(4,4)->(4,5)->(5,5)
# ---end_answer---


# ---start_reasoning---
# No valid path exists from start to end.
# ---end_reasoning---

# ---start_answer---
# not exist
# ---end_answer---
##