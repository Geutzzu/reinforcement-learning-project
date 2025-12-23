import argparse
import torch
import sys
sys.path.append("..")
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from config_loader import load_config
from utils import load_datasets
from experiments import ExperimentTracker


def main(config_path: str):
    config = load_config(config_path, "sft")

    if torch.cuda.is_available():
        print("Using GPU")
    if torch.backends.mps.is_available():
        print("Using MPS")

    print(f"Task: {config.reward_fn}")
    print(f"Model: {config.model_name}")

    tracker = ExperimentTracker(
        name=config.experiment_name or "experiment",
        algorithm="sft",
        output_dir=config.output_dir,
    )
    print(f"Run: {tracker.run_id}")

    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    train_dataset, eval_dataset = load_datasets(config)

    model_kwargs = {"attn_implementation": "flash_attention_2"} if torch.cuda.is_available() else {}

    sft_config = SFTConfig(
        output_dir=tracker.run_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        max_length=config.max_length,
        save_steps=config.save_steps,
        bf16=torch.cuda.is_available() or torch.backends.mps.is_available(),
        report_to="none",
        use_liger_kernel=config.use_liger_kernel and torch.cuda.is_available(),
        model_init_kwargs=model_kwargs,
    )

    trainer = SFTTrainer(
        model=config.model_name,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[tracker.get_callback()],
    )

    trainer.train()
    trainer.save_model()
    trainer.processing_class.save_pretrained(tracker.run_dir)
    print(f"Saved to {tracker.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
