import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from main.enigmata import make_reward_fns

IS_MPS = torch.backends.mps.is_available()

if IS_MPS:
    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        HAS_MLX = True
        print("[Evaluate] Using MLX for fast Mac inference")
    except ImportError:
        HAS_MLX = False
        print("[Evaluate] MLX not installed. Run: pip install mlx mlx-lm")
else:
    HAS_MLX = False


def load_model(model_path: str):
    if IS_MPS and HAS_MLX:
        model, tokenizer = mlx_load(model_path)
        return model, tokenizer, "mlx"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer, "torch"


def generate_predictions(model, tokenizer, prompts: list[str], backend: str = "torch", max_new_tokens: int = 512, temperature: float = 0.0) -> list[str]:
    predictions = []
    
    if backend == "mlx":
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        for prompt in tqdm(prompts, desc="Generating (MLX)"):
            response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, sampler=sampler)
            predictions.append(response)
    else:
        model.eval()
        for prompt in tqdm(prompts, desc="Generating (PyTorch)"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature or None, do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id)
            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.append(pred)
    
    return predictions


def evaluate_on_dataset(model_path: str, test_path: str, task: str) -> dict:
    model, tokenizer, backend = load_model(model_path)
    df = pd.read_parquet(test_path)
    df = df.sample(20, random_state=42)
    
    predictions = generate_predictions(model, tokenizer, df["prompt"].tolist(), backend=backend)
    df["prediction"] = predictions
    
    reward_fns = make_reward_fns([task])
    rewards = reward_fns[0](df["prompt"].tolist(), predictions, answer=df["answer"].tolist(), meta=df["meta"].tolist())
    df["reward"] = rewards
    
    accuracy = sum(r > 0 for r in rewards) / len(rewards)
    return {
        "accuracy": accuracy,
        "mean_reward": sum(rewards) / len(rewards),
        "total": len(rewards),
        "correct": sum(r > 0 for r in rewards),
        "predictions_df": df
    }


def evaluate_model(model_path: str, test_paths: dict[str, str]) -> dict:
    results = {}
    for task, path in test_paths.items():
        results[task] = evaluate_on_dataset(model_path, path, task)
        results[task].pop("predictions_df")  # Don't include in summary
    return results



if __name__ == "__main__":
    evaluate_model("/Users/geo/facultate/rl/rl/results/maze_2_grpo_quick/3B_pure_rl_with_one_correct_answer/merged", {"maze": "/Users/geo/facultate/rl/rl/data/maze_2_reasoning.parquet"})
