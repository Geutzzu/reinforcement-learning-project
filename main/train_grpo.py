import argparse
import torch
import sys
sys.path.append("..")
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from config_loader import load_config
from enigmata import make_reward_fn, get_verifier
from utils import load_datasets
from experiments import ExperimentTracker


def main(config_path: str):
    config = load_config(config_path, "grpo")

    if torch.cuda.is_available():
        print("Using GPU")
    if torch.backends.mps.is_available():
        print("Using MPS")

    print(f"Reward: {config.reward_fn}")
    print(f"Model: {config.model_name}")
    
    if config.use_vllm:
        print(f"vLLM: mode={config.vllm_mode}, gpu_mem={config.vllm_gpu_memory_utilization}")

    from dataclasses import asdict
    tracker = ExperimentTracker(
        name=config.experiment_name or "experiment",
        algorithm="grpo",
        output_dir=config.output_dir,
        config=asdict(config),
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
    reward_fn = make_reward_fn(config.reward_fn)
    
    verifier = get_verifier(config.reward_fn)
    tracker.save_reward_fn(verifier, name=config.reward_fn)

    callbacks = [tracker.get_callback()]
    if config.snapshot_prompts_count > 0:
        from experiments import SnapshotCallback
        sample_prompts = [train_dataset[i]["prompt"] for i in range(min(config.snapshot_prompts_count, len(train_dataset)))]
        callbacks.append(SnapshotCallback(sample_prompts, tracker.run_dir, snapshot_every_n_steps=config.snapshot_every_n_steps))

    model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {} # to be uncommented after flash-attn is installed

    grpo_config = GRPOConfig(
        output_dir=tracker.run_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.num_generations,  # Must be divisible by num_generations
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        eval_strategy=config.eval_strategy if eval_dataset is not None else "no",
        eval_steps=config.eval_steps,
        bf16=torch.cuda.is_available() or torch.backends.mps.is_available(),
        report_to="none",
        log_completions=config.log_completions,
        use_liger_kernel=config.use_liger_kernel and torch.cuda.is_available(),
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        model_init_kwargs=model_kwargs,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        vllm_server_host=config.vllm_server_host,
        vllm_server_port=config.vllm_server_port,
        vllm_server_timeout=config.vllm_server_timeout,
    )

    trainer = GRPOTrainer(
        model=config.model_name,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        callbacks=callbacks,
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




# /venv/main