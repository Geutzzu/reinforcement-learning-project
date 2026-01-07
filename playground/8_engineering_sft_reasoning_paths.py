
import json
import re
import sys
sys.path.append("/Users/geo/facultate/rl/rl")
from numpy import sort_complex
import pandas as pd
from Enigmata.verifiable_tasks.tasks.maze_2.verifier import verify, verify_format, verify_answer


df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/10k_preds_with_1e_sft_base_and_grpo.parquet")

df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze_2_reasoning_new_sample_new_template.parquet")


