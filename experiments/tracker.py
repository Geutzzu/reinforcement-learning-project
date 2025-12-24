import os
import json
import torch
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
    config: dict = None
    logs: list = field(default_factory=list)
    
    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.run_id = f"{timestamp}_{self.algorithm}_{self.name}"
        self.run_dir = os.path.join(self.output_dir, self.name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._save_config()
        self._setup_logging()
    
    def _setup_logging(self):
        import sys
        log_path = os.path.join(self.run_dir, "training.log")
        self.log_file = open(log_path, "w")
        
        class TeeStream:
            def __init__(self, original, file):
                self.original = original
                self.file = file
            def write(self, data):
                self.original.write(data)
                self.file.write(data)
                self.file.flush()
            def flush(self):
                self.original.flush()
                self.file.flush()
        
        sys.stdout = TeeStream(sys.stdout, self.log_file)
        sys.stderr = TeeStream(sys.stderr, self.log_file)
    
    def _save_config(self):
        data = {"name": self.name, "algorithm": self.algorithm, "run_id": self.run_id}
        if self.config:
            data["training_config"] = self.config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(data, f, indent=2)
    
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
        self.save_every_n_logs = 10
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.tracker.log(logs, step=state.global_step)
            if len(self.tracker.logs) % self.save_every_n_logs == 0:
                self.tracker.save()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.tracker.save()
    
    def on_train_end(self, args, state, control, **kwargs):
        self.tracker.save()


class SnapshotCallback(TrainerCallback):
    def __init__(self, sample_prompts: list[str], output_dir: str, max_new_tokens: int = 256, snapshot_every_n_steps: int = None):
        self.sample_prompts = sample_prompts
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.snapshot_every_n_steps = snapshot_every_n_steps
        self.snapshots = []
        os.makedirs(output_dir, exist_ok=True)
    
    def _capture(self, model, tokenizer, step: int, epoch: int = None):
        model.eval()
        entry = {"step": step, "epoch": epoch, "predictions": []}
        
        for prompt in self.sample_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            entry["predictions"].append({"prompt": prompt[:100], "response": pred})
        
        self.snapshots.append(entry)
        with open(os.path.join(self.output_dir, "snapshots.json"), "w") as f:
            json.dump(self.snapshots, f, indent=2)
        model.train()
    
    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.snapshot_every_n_steps and model and processing_class:
            if state.global_step % self.snapshot_every_n_steps == 0:
                self._capture(model, processing_class, state.global_step)
    
    def on_epoch_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if not self.snapshot_every_n_steps and model and processing_class:
            self._capture(model, processing_class, state.global_step, int(state.epoch))
