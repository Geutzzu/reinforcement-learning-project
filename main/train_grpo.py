"""
GRPO Training Script for Enigmata Puzzles.
Uses TRL's GRPOTrainer with Enigmata task reward functions.

Usage:
    python train_grpo.py --task sudoku2
    python train_grpo.py --task maze --difficulty hard
    python train_grpo.py --task sudoku2 --smoke_test
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from config import TrainingConfig
from enigmata import generate_puzzles, make_reward_fn, AVAILABLE_TASKS


def main(args):
    config = TrainingConfig()

    # Override with CLI args
    task_name = args.task or config.puzzle_task
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.difficulty:
        config.difficulty = args.difficulty
    if args.smoke_test:
        config.max_steps = 10
        config.num_train_samples = 50
        config.batch_size = 2

    print(f"Task: {task_name}")
    print(f"Available tasks: {len(AVAILABLE_TASKS)}")
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Apply LoRA
    if config.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset using Enigmata
    print(f"Generating {config.num_train_samples} {task_name} puzzles...")
    train_dataset = generate_puzzles(
        task_name,
        count=config.num_train_samples,
        difficulty=getattr(config, "difficulty", "medium"),
    )

    # Get reward function
    reward_fn = make_reward_fn(task_name)

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=f"{config.output_dir}/{task_name}",
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Create trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    output_path = f"{config.output_dir}/{task_name}"
    print(f"Saving model to {output_path}")
    trainer.save_model()
    tokenizer.save_pretrained(output_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=f"Task name. Available: {AVAILABLE_TASKS[:5]}...",
    )
    parser.add_argument(
        "--difficulty", type=str, default=None, choices=["easy", "medium", "hard"]
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Quick test run")
    args = parser.parse_args()
    main(args)
