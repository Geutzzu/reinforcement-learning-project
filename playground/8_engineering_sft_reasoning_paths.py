
import json
import re
import sys
sys.path.append("/Users/geo/facultate/rl/rl")
from numpy import sort_complex
import pandas as pd
from Enigmata.verifiable_tasks.tasks.maze_2.verifier import verify, verify_format, verify_answer


df1 = pd.read_parquet("/Users/geo/facultate/rl/rl/data/rl_maze_2_exhaustive_reasoning.parquet")
df2 = pd.read_parquet("/Users/geo/facultate/rl/rl/data/sft_maze_2_exhaustive_reasoning.parquet")

df_to_reject = pd.concat([df1, df2]).drop_duplicates(subset=["prompt"]).reset_index(drop=True)


from main.enigmata import generate_puzzles
from utils.processing_tools import batch_tokenize

df1 = generate_puzzles("maze_2", count=4500, difficulty="easy")
df2 = generate_puzzles("maze_2", count=4500, difficulty="medium")
df3 = generate_puzzles("maze_2", count=4500, difficulty="hard")
df = pd.concat([df1, df2, df3], ignore_index=True)
df['tokens_len'] = batch_tokenize(df.apply(lambda x: x['prompt'] + "\n\n" + x['answer'], axis=1))



df = df[~df['prompt'].isin(df_to_reject['prompt'])].reset_index(drop=True)
df.to_parquet("/Users/geo/facultate/rl/rl/data/test_maze_2_exhaustive_reasoning.parquet", index=False)
