import sys
import os
import importlib
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from datasets import Dataset


ENIGMATA_PATH = "/Users/geo/facultate/rl/rl/Enigmata"

if ENIGMATA_PATH not in sys.path:
    sys.path.insert(0, ENIGMATA_PATH)

TASKS_DIR = os.path.join(ENIGMATA_PATH, "verifiable_tasks", "tasks")

def _discover_tasks() -> Dict[str, Dict[str, Any]]:
    tasks = {}
    for task_name in sorted(os.listdir(TASKS_DIR)):
        task_dir = os.path.join(TASKS_DIR, task_name)
        if not os.path.isdir(task_dir):
            continue
        if not os.path.exists(os.path.join(task_dir, "generator.py")):
            continue
        
        try:
            gen_module = importlib.import_module(f"verifiable_tasks.tasks.{task_name}.generator")
            ver_module = importlib.import_module(f"verifiable_tasks.tasks.{task_name}.verifier")
            tasks[task_name] = {
                "generate": gen_module.generate,
                "verify": ver_module.verify,
            }
        except Exception:
            print(f"Failed to load task {task_name}")
            pass
    
    return tasks

TASKS = _discover_tasks()
AVAILABLE_TASKS = list(TASKS.keys())


def generate_puzzles(task_name: str, count: int = 100, difficulty: str = "medium") -> pd.DataFrame:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {AVAILABLE_TASKS}")
    
    generate_fn = TASKS[task_name]["generate"]

    puzzles = []
    
    for puzzle in generate_fn(count=count, difficulty=difficulty, language="en", split="train"):
        puzzle["task_name"] = task_name
        puzzles.append(puzzle)
        if len(puzzles) >= count:
            break
    
    return pd.DataFrame(puzzles)



def generate_mixed(task_counts: Dict[str, int], difficulty: str = "medium", shuffle: bool = True) -> pd.DataFrame:
    dfs = []
    
    for task_name, count in task_counts.items():
        df = generate_puzzles(task_name, count=count, difficulty=difficulty)
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    
    if shuffle:
        result = result.sample(frac=1).reset_index(drop=True)
    
    return result


def verify(task_name: str, pred: str, answer: Any, meta: Any) -> int:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASKS[task_name]["verify"](pred, answer, meta)


def make_reward_fn(default_task: str = None) -> Callable:

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        rewards = []
        n = len(completions)
        
        answer = kwargs["answer"]
        meta = kwargs["meta"]
        task_name_list = kwargs.get("task_name") or [default_task] * n
        
        for comp, ans, m, t in zip(completions, answer, meta, task_name_list):
            try:
                t = t or default_task
                if t and t in TASKS:
                    rewards.append(float(TASKS[t]["verify"](comp, ans, m)))
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        
        return rewards
    
    return reward_fn


def get_verifier(task_name: str) -> Callable:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {AVAILABLE_TASKS}")
    return TASKS[task_name]["verify"]


def to_hf_dataset(df: pd.DataFrame):
    df = df.rename(columns={"answer": "completion"})
    return Dataset.from_pandas(df)

        
if __name__ == "__main__":
    print(f"Loaded {len(AVAILABLE_TASKS)} tasks: {AVAILABLE_TASKS}")
    
    df = generate_puzzles("sudoku2", count=3, difficulty="easy")
    print(f"\nSingle task shape: {df.shape}")
    print(df[["task_name", "prompt"]].head())
    
    df_mixed = generate_mixed({"sudoku2": 2, "maze": 2}, difficulty="easy")
    print(f"\nMixed dataset shape: {df_mixed.shape}")
    print(df_mixed[["task_name"]].value_counts())

