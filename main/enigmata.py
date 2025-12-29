import sys
import os
import importlib
import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Tuple
from datasets import Dataset


ENIGMATA_PATH = "/workspace/rl/Enigmata"

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
            
            reward_fns = getattr(ver_module, "REWARD_FUNCTIONS", {"verify": ver_module.verify})
            
            tasks[task_name] = {
                "generate": gen_module.generate,
                "reward_functions": reward_fns,
            }
        except Exception:
            print(f"Failed to load task {task_name}")
    
    return tasks


TASKS = _discover_tasks()
AVAILABLE_TASKS = list(TASKS.keys())


def parse_reward_fn_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        task_name, fn_name = spec.split(":", 1)
    else:
        task_name, fn_name = spec, "verify"
    return task_name, fn_name


def get_reward_fn(task_name: str, fn_name: str = "verify") -> Callable:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {AVAILABLE_TASKS}")
    
    available = TASKS[task_name]["reward_functions"]
    if fn_name not in available:
        raise ValueError(f"Unknown reward function '{fn_name}' for task '{task_name}'. Available: {list(available.keys())}")
    
    return available[fn_name]


def get_available_reward_fns(task_name: str) -> List[str]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {AVAILABLE_TASKS}")
    return list(TASKS[task_name]["reward_functions"].keys())


def make_reward_fns(reward_fn_specs: List[str]) -> List[Callable]:
    def _wrap_for_trl(vfn: Callable) -> Callable:
        def wrapped(prompts, completions, **kwargs) -> List[float]:
            rewards = []
            for comp, ans, m in zip(completions, kwargs["answer"], kwargs["meta"]):
                try:
                    rewards.append(float(vfn(comp, ans, m)))
                except Exception:
                    rewards.append(0.0)
            return rewards
        return wrapped
    
    reward_funcs = []
    for spec in reward_fn_specs:
        task_name, fn_name = parse_reward_fn_spec(spec)
        verify_fn = get_reward_fn(task_name, fn_name)
        reward_funcs.append(_wrap_for_trl(verify_fn))
    
    return reward_funcs


def get_task_from_reward_fns(reward_fn_specs: List[str]) -> str:
    if not reward_fn_specs:
        raise ValueError("reward_fns cannot be empty")
    task_name, _ = parse_reward_fn_spec(reward_fn_specs[0])
    return task_name



def get_verifier(task_name: str) -> Callable:
    return get_reward_fn(task_name, "verify")


def verify(task_name: str, pred: str, answer: Any, meta: Any) -> float:
    return get_reward_fn(task_name, "verify")(pred, answer, meta)




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
    dfs = [generate_puzzles(task_name, count=count, difficulty=difficulty) 
           for task_name, count in task_counts.items()]
    
    result = pd.concat(dfs, ignore_index=True)
    if shuffle:
        result = result.sample(frac=1).reset_index(drop=True)
    return result


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
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
