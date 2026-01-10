import sys
import pandas as pd

sys.path.insert(0, "/Users/geo/facultate/rl/rl")

from main.enigmata import generate_puzzles
from utils.processing_tools import batch_tokenize

df1 = generate_puzzles("maze_2", count=9000, difficulty="easy")
df2 = generate_puzzles("maze_2", count=9000, difficulty="medium")
df3 = generate_puzzles("maze_2", count=9000, difficulty="hard")
df = pd.concat([df1, df2, df3], ignore_index=True)
df['tokens_len'] = batch_tokenize(df.apply(lambda x: x['prompt'] + "\n\n" + x['answer'], axis=1))

solvable = df[df['answer'].str.contains("not exist") == False]
unsolvable = df[df['answer'].str.contains("not exist")]
print(f"Solvable: {len(solvable)}, Unsolvable: {len(unsolvable)}")

n_unsolvable_target = int(len(solvable) * 0.15)
unsolvable_sampled = unsolvable.sample(n=min(n_unsolvable_target, len(unsolvable)))
df_balanced = pd.concat([solvable, unsolvable_sampled]).sample(frac=1).reset_index(drop=True)
print(f"Balanced: {len(df_balanced)} total")

df_balanced['answer_block'] = df_balanced['answer'].apply(lambda x: x.split("---start_answer---", 1)[1].split("---end_answer---", 1)[0].strip())
df_balanced = df_balanced.drop_duplicates(subset=["prompt", "answer_block"]).reset_index(drop=True)
print(f"After deduplication: {len(df_balanced)}")

df_sft = df_balanced.sample(frac=0.5)
df_rl = df_balanced.drop(index=df_sft.index)

df_balanced['answer_tokens_len'] = batch_tokenize(df_balanced.apply(lambda x: x['answer'], axis=1))

df_sft.to_parquet("/Users/geo/facultate/rl/rl/data/sft_maze_2_exhaustive_reasoning.parquet") # ~1300 max total tokens
df_rl.to_parquet("/Users/geo/facultate/rl/rl/data/rl_maze_2_exhaustive_reasoning.parquet") # ~956.0 max answer tokens -> 1024 max should work ideally, 

df_balanced['answer_tokens_len'].describe() # 1267.0 max total almost likelly 1024 






## letter count

import sys
import pandas as pd

sys.path.insert(0, "/Users/geo/facultate/rl/rl")

from main.enigmata import generate_puzzles
from utils.processing_tools import batch_tokenize

df1 = generate_puzzles("letter_count", count=7000, difficulty="easy")
df2 = generate_puzzles("letter_count", count=7000, difficulty="medium")
df3 = generate_puzzles("letter_count", count=7000, difficulty="hard")
df = pd.concat([df1, df2, df3], ignore_index=True)
df['tokens_len'] = batch_tokenize(df.apply(lambda x: x['prompt'] + "\n\n" + x['answer'], axis=1))

df_sft = df.sample(frac=0.5)
df_rl = df.drop(index=df_sft.index)

df['answer_tokens_len'] = batch_tokenize(df.apply(lambda x: x['answer'], axis=1))

df_sft.to_parquet("/Users/geo/facultate/rl/rl/data/sft_letter_count.parquet") # ~1300 max total tokens
df_rl.to_parquet("/Users/geo/facultate/rl/rl/data/rl_letter_count.parquet") # ~956.0 max answer tokens -> 1024 max should work ideally, \


