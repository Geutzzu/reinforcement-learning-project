from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training
    max_steps: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    
    # GRPO specific
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Output
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 100
    
    # Puzzle config
    puzzle_task: str = "sudoku2"
    difficulty: str = "medium"
    num_train_samples: int = 2000
    num_eval_samples: int = 200



@dataclass
class SFTConfig:
    """Config for supervised fine-tuning baseline."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    max_steps: int = 500
    batch_size: int = 4
    learning_rate: float = 2e-5
    output_dir: str = "./outputs_sft"
