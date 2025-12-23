import argparse
import torch
import sys
sys.path.append("..")
from trl.experimental.ppo import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from config_loader import load_config
from enigmata import make_reward_fn
from utils import load_datasets
from experiments import ExperimentTracker
from dataclasses import asdict


class RewardWrapper(torch.nn.Module):
    """Wraps a reward function to look like a reward model for PPO."""
    
    def __init__(self, reward_fn, tokenizer):
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        # Dummy parameter so it's a valid Module
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Decode and score each sequence
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Extract prompts and completions (assuming format: prompt + completion)
        rewards = []
        for text in texts:
            # The reward_fn expects (prompts, completions) format
            # For now, treat whole text as completion and use dummy prompt
            reward = self.reward_fn([text], [text])[0]
            rewards.append(reward)
        
        return torch.tensor(rewards, device=input_ids.device, dtype=torch.float32)


def main(config_path: str):
    config = load_config(config_path, "grpo")

    if torch.cuda.is_available():
        print("Using GPU")
    if torch.backends.mps.is_available():
        print("Using MPS")

    print(f"Reward: {config.reward_fn}")
    print(f"Model: {config.model_name}")

    tracker = ExperimentTracker(
        name=config.experiment_name or "experiment",
        algorithm="ppo",
        output_dir=config.output_dir,
        config=asdict(config),
    )
    print(f"Run: {tracker.run_id}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)

    reward_fn = make_reward_fn(config.reward_fn)
    reward_model = RewardWrapper(reward_fn, tokenizer)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    train_dataset, eval_dataset = load_datasets(config)

    ppo_config = PPOConfig(
        output_dir=tracker.run_dir,
        total_episodes=config.max_steps if config.max_steps > 0 else 1000,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        temperature=config.temperature,
        response_length=config.max_new_tokens,
        num_ppo_epochs=4,
        kl_coef=config.beta,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[tracker.get_callback()],
    )

    trainer.train()
    trainer.save_model(tracker.run_dir)
    tokenizer.save_pretrained(tracker.run_dir)
    print(f"Saved to {tracker.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
