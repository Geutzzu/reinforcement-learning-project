from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BaseConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    num_epochs: int = 3
    max_steps: int = -1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"  # Options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    warmup_ratio: float = 0.03  # 3% warmup recommended
    max_length: int = 512
    use_liger_kernel: bool = False
    optim: str = "adamw_torch_fused"  # Options: adamw_torch_fused, adamw_8bit
    gradient_checkpointing: bool = True  # Trades compute for memory
    
    output_dir: str = "/Users/geo/facultate/rl/rl/results"
    logging_steps: int = 10
    save_steps: int = 100
    
    train_dataset_path: str = None
    eval_dataset_path: str = None
    eval_split_ratio: float = 0.0
    
    # Evaluation settings
    eval_strategy: str = "epoch"  # Options: "no", "epoch", "steps"
    eval_steps: int = None  # Only used if eval_strategy="steps"
    
    experiment_name: str = None
    
    # Reward functions: ["task:function", ...] e.g. ["maze_2:verify"] or ["maze_2:format", "maze_2:path_solved"]
    reward_fns: List[str] = field(default_factory=lambda: ["maze:verify"])
    reward_weights: List[float] = field(default_factory=list)  # Optional weights (must match reward_fns length)
    
    # Snapshot settings
    snapshot_prompts_count: int = 0  # Number of prompts to sample for snapshots (0 = disabled)
    snapshot_every_n_steps: int = None  # If set, snapshot every N steps; else per epoch


@dataclass
class SFTConfig(BaseConfig):
    learning_rate: float = 2e-5


@dataclass
class GRPOConfig(BaseConfig):
    learning_rate: float = 1e-5
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    beta: float = 0.0  # KL penalty coefficient (0.0 = no KL penalty, recommended by recent papers)
    
    # Loss type: "grpo", "dapo", "dr_grpo", "sapo", "bnpo", "cispo"
    loss_type: str = "dapo"  # DAPO recommended for long-CoT, no length bias
    scale_rewards: str = "batch"  # "group", "batch", or "none" - batch is more robust (PPO Lite)
    
    # SAPO-specific (only used when loss_type="sapo")
    sapo_temperature_neg: float = 1.05  # Temperature for negative advantages
    sapo_temperature_pos: float = 1.0   # Temperature for positive advantages
    
    # Logging
    log_completions: bool = False  # Log training generations (requires 'rich' package)
    
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    vllm_enable_sleep_mode: bool = False
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_max_model_length: int = None  # Max context length for vLLM (None = use model default)
    
    # Replay buffer (experimental)
    replay_buffer_size: int = 64  # Size of replay buffer for storing high-reward rollouts
    
    # LLDS (Lazy Likelihood Displacement Suppression) - prevents GRPO collapse
    # From: "ON GRPO COLLAPSE IN SEARCH-R1: THE LAZY LIKELIHOOD-DISPLACEMENT DEATH SPIRAL"
    llds_enabled: bool = False  # Enable LLDS regularization
    llds_lambda: float = 0.1    # Regularization coefficient (paper default = 0.1)
    llds_start_step: int = 0    # Step to start applying LLDS (0 = from beginning)

