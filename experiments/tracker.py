import os
import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from transformers import TrainerCallback


@dataclass
class ExperimentTracker:
    name: str
    algorithm: str
    output_dir: str = "results"
    logs: list = field(default_factory=list)
    
    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.algorithm}_{self.name}_{timestamp}"
        self.run_dir = os.path.join(self.output_dir, self.name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._save_config()
    
    def _save_config(self):
        config = {"name": self.name, "algorithm": self.algorithm, "run_id": self.run_id}
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    def log(self, metrics: dict[str, Any], step: int = None):
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        self.logs.append(entry)
    
    def save(self):
        df = pd.DataFrame(self.logs)
        df.to_parquet(os.path.join(self.run_dir, "logs.parquet"), index=False)
        df.to_csv(os.path.join(self.run_dir, "logs.csv"), index=False)
    
    def save_eval_results(self, results: dict):
        with open(os.path.join(self.run_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    def get_callback(self):
        return TrackerCallback(self)


class TrackerCallback(TrainerCallback):
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.tracker.log(logs, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.tracker.save()
