import sys
import pandas as pd 

sys.path.insert(0, "/workspace/rl")

from main.enigmata import generate_puzzles
# from utils.processing_tools import batch_tokenize

df = generate_puzzles("maze_2", count=22500, difficulty="hard")
# df['tokens_len'] = batch_tokenize(df['answer'])

solvable = df[df['answer'].str.contains("not exist") == False]
unsolvable = df[df['answer'].str.contains("not exist")]
print(f"Solvable: {len(solvable)}, Unsolvable: {len(unsolvable)}")

n_unsolvable_target = int(len(solvable) * 0.25)
unsolvable_sampled = unsolvable.sample(n=min(n_unsolvable_target, len(unsolvable)))
df_balanced = pd.concat([solvable, unsolvable_sampled]).sample(frac=1).reset_index(drop=True)
print(f"Balanced: {len(df_balanced)} total")

not_include = pd.read_parquet("/workspace/rl/data/maze_2_reasoning.parquet")
df_balanced = df_balanced[~df_balanced['prompt'].isin(not_include['prompt'])]

df_balanced.to_parquet("/workspace/rl/data/maze_2_reasoning_new_sample_new_template.parquet")

