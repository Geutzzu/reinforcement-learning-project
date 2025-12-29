import sys
import pandas as pd 

sys.path.insert(0, "/Users/geo/facultate/rl/rl")


from main.enigmata import generate_puzzles, verify, AVAILABLE_TASKS
from utils.processing_tools import batch_tokenize
from Enigmata.verifiable_tasks.tasks.maze.verifier import verify
from utils.gemini_utils import GeminiLLM

llm = GeminiLLM(model="gemini-2.5-flash")

test_df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze_2_reasoning.parquet")
test_df = test_df.sample(30, random_state=42)

test_df['gemini_pred'] = llm.predict_batch(test_df['prompt'].tolist(), max_output_tokens=12000)

test_df['gemini_answer_block'] = test_df['gemini_pred'].apply(lambda x: x.split('---start_answer---')[1].split('---end_answer---')[0] if '---start_answer---' in x and '---end_answer---' in x else "asdsa")
test_df['answer_block'] = test_df['answer'].apply(lambda x: x.split('---start_answer---')[1].split('---end_answer---')[0])

test_df['is_correct'] = test_df.apply(lambda x: verify(x['gemini_answer_block'], x['answer_block'], x['meta']), axis=1)

test_df['answer'] = test_df['answer'].apply(lambda x: x.strip())
test_df['meta'] = test_df['meta'].apply(lambda x: json.loads(x))
test_df['pred'] = test_df['pred'].apply(lambda x: x.strip())

test_df['reward'] = test_df.apply(lambda x: verify(x['pred'], x['answer'], x['meta']), axis=1)