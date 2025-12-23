import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from enigmata import make_reward_fn


def load_model(model_path: str, base_model: str = None):
    if base_model:  # LoRA adapter
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, model_path)
    else:  # Full model
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path if not base_model else base_model)
    return model, tokenizer


def generate_predictions(model, tokenizer, prompts: list[str], max_new_tokens: int = 512, temperature: float = 0.0) -> list[str]:
    model.eval()
    predictions = []
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature or None, do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id)
        pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(pred)
    return predictions


def evaluate_on_dataset(model_path: str, test_path: str, task: str, base_model: str = None) -> dict:
    model, tokenizer = load_model(model_path, base_model)
    df = pd.read_parquet(test_path)
    
    predictions = generate_predictions(model, tokenizer, df["prompt"].tolist())
    df["prediction"] = predictions
    
    reward_fn = make_reward_fn(task)
    rewards = reward_fn(df["prompt"].tolist(), predictions)
    df["reward"] = rewards
    
    accuracy = sum(r > 0 for r in rewards) / len(rewards)
    return {
        "accuracy": accuracy,
        "mean_reward": sum(rewards) / len(rewards),
        "total": len(rewards),
        "correct": sum(r > 0 for r in rewards),
        "predictions_df": df
    }


def evaluate_model(model_path: str, test_paths: dict[str, str], base_model: str = None) -> dict:
    """Evaluate on multiple test sets. test_paths = {"maze": "data/maze/test.parquet", ...}"""
    results = {}
    for task, path in test_paths.items():
        results[task] = evaluate_on_dataset(model_path, path, task, base_model)
        results[task].pop("predictions_df")  # Don't include in summary
    return results
