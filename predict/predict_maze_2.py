import pandas as pd
from utils.vllm_utils import VLLM

test_df = pd.read_parquet("/workspace/rl/data/test_maze_2_exhaustive_reasoning.parquet")

MODEL_PATH = "/workspace/rl/models/0.5B-extended-reasoning-1e"
MODEL_PATH_FULL_TRAIN = "/workspace/rl/models/0.5B-extended-reasoning-1e-plus-1000-grpo-steps-0.01kl-dr_dapo"
OUTPUT_PATH = "/workspace/rl/data/test_maze_2_exhaustive_reasoning.parquet"


if __name__ == "__main__":
    vllm = VLLM(MODEL_PATH)

    test_df['grpo_pred'] = vllm.predict_batch(test_df['prompt'].tolist())

    test_df.to_parquet(OUTPUT_PATH)




# import sys
# import pandas as pd 

# sys.path.insert(0, "/Users/geo/facultate/rl/rl")

# from Enigmata.verifiable_tasks.tasks.maze_2.verifier import verify
# # from utils.vllm_utils import VLLM

# df = pd.read_parquet("/workspace/rl/data/10k_preds_with_1e_sft_base_and_grpo.parquet")

# sft_rewards = []
# grpo_rewards = []

# for idx, row in df.iterrows():
#     sft_r = verify(row['base_sft_pred'], row['answer'], row['meta'])
#     grpo_r = verify(row['grpo_pred'], row['answer'], row['meta'])
#     sft_rewards.append(sft_r)
#     grpo_rewards.append(grpo_r)
#     if int(idx) % 1000 == 0:
#         print(f'Processed {idx}/{len(df)}...')

# df['sft_reward'] = sft_rewards
# df['grpo_reward'] = grpo_rewards

# sft_acc = (df['sft_reward'] >= 1.5).mean()
# grpo_acc = (df['grpo_reward'] >= 1.5).mean()

# print()
# print('=' * 50)
# print('ACCURACY COMPARISON (reward >= 1.5 = success)')
# print('=' * 50)
# print(f'SFT Base Accuracy:  {sft_acc:.4f} ({(df["sft_reward"] >= 1.5).sum()}/{len(df)})')
# print(f'GRPO Accuracy:      {grpo_acc:.4f} ({(df["grpo_reward"] >= 1.5).sum()}/{len(df)})')
# print(f'Improvement:        {(grpo_acc - sft_acc)*100:+.2f}%')
# print()
# print('REWARD STATISTICS')
# print('=' * 50)
# print(f'SFT Mean Reward:  {df["sft_reward"].mean():.4f}')
# print(f'GRPO Mean Reward: {df["grpo_reward"].mean():.4f}')
# print()
# print('REWARD DISTRIBUTION')
# print('SFT:')
# print(df['sft_reward'].describe())
# print()
# print('GRPO:')
# print(df['grpo_reward'].describe())

# print()
# print('=' * 50)
# print('EXAMPLES WHERE GRPO IMPROVED (SFT failed, GRPO succeeded)')
# print('=' * 50)
# improved = df[(df['sft_reward'] < 1.5) & (df['grpo_reward'] >= 1.5)]
# print(f'Count: {len(improved)}')

# print()
# print('=' * 50)
# print('EXAMPLES WHERE GRPO DEGRADED (SFT succeeded, GRPO failed)')
# print('=' * 50)
# degraded = df[(df['sft_reward'] >= 1.5) & (df['grpo_reward'] < 1.5)]
# print(f'Count: {len(degraded)}')


