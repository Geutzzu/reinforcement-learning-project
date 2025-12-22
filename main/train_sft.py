import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from config import SFTConfig as MySFTConfig
from enigmata import generate_puzzles, AVAILABLE_TASKS


def format_for_sft(example):
    answer = example["answer"]
    return {"text": f"{example['prompt']}\n{answer}"}


def main(args):
    config = MySFTConfig()
    task_name = args.task or "sudoku2"

    if args.max_steps:
        config.max_steps = args.max_steps
    if args.smoke_test:
        config.max_steps = 10

    print(f"Task: {task_name}")
    print(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load dataset using Enigmata
    print(f"Generating dataset for {task_name}...")
    train_dataset = generate_puzzles(task_name, count=1000, difficulty="medium")
    train_dataset = train_dataset.map(format_for_sft)

    # LoRA config
    lora_config = None
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    # SFT Config
    sft_config = SFTConfig(
        output_dir=f"{config.output_dir}/{task_name}",
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Create trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting SFT training...")
    trainer.train()

    output_path = f"{config.output_dir}/{task_name}"
    print(f"Saving to {output_path}")
    trainer.save_model()
    tokenizer.save_pretrained(output_path)

    print("SFT training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help=f"Task name")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    main(args)
