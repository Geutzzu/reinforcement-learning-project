"""
PPO Trainer with Custom Reward Functions

This module provides a PPOTrainer subclass that supports custom reward functions
(like GRPO does) instead of requiring a neural reward model.

Based on the approach from PR #2540: https://github.com/huggingface/trl/issues/2540

The key insight is that TRL's PPO uses `get_reward()` from `trl.experimental.utils`
to compute rewards. We override the training step to inject our custom rewards.
"""

import argparse
import torch
import torch.nn as nn
import sys
import gc
import numpy as np
from contextlib import nullcontext

sys.path.append("..")

from trl.experimental.ppo import PPOTrainer as BasePPOTrainer, PPOConfig
from trl.experimental.ppo.ppo_trainer import (
    AutoModelForCausalLMWithValueHead,
    batch_generation,
    generate,
    INVALID_LOGPROB,
)
from trl.trainer.utils import (
    disable_dropout_in_model,
    empty_cache,
    pad,
    selective_log_softmax,
)
from trl.experimental.utils import first_true_indices, get_reward
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, GenerationConfig
from accelerate.utils import broadcast, gather_object
from config_loader import load_config
from enigmata import make_reward_fn, get_verifier
from utils import load_datasets
from experiments import ExperimentTracker
from dataclasses import asdict


class DummyRewardModel(nn.Module):
    """
    Minimal reward model that satisfies PPOTrainer's interface requirements.
    The actual reward computation is done by the custom reward function.
    """
    base_model_prefix = "model"
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.model = nn.Linear(hidden_size, hidden_size)
        self._hidden_size = hidden_size
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        hidden = torch.zeros(batch_size, 1, self._hidden_size, device=input_ids.device)
        return type('Output', (), {'hidden_states': [hidden]})()
    
    def score(self, hidden_states):
        return torch.zeros(hidden_states.shape[0], 1, device=hidden_states.device)


class CustomRewardPPOTrainer(BasePPOTrainer):
    """
    PPOTrainer that supports custom reward functions instead of neural reward models.
    
    The reward function should have the signature:
        def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]
    
    This matches the GRPO reward function interface for easy compatibility.
    """
    
    def __init__(
        self, 
        *args, 
        reward_func=None,
        **kwargs
    ):
        """
        Args:
            reward_func: A callable that takes (prompts, completions) and returns rewards.
                        If None, uses the reward_model as in standard PPO.
            *args, **kwargs: Passed to BasePPOTrainer.
        """
        self.custom_reward_func = reward_func
        
        # If using custom reward function, provide a dummy reward model
        if reward_func is not None:
            if 'reward_model' not in kwargs or kwargs['reward_model'] is None:
                policy_model = kwargs.get('model') or (args[2] if len(args) > 2 else None)
                hidden_size = 768
                if policy_model is not None and hasattr(policy_model, 'config'):
                    hidden_size = getattr(policy_model.config, 'hidden_size', 768)
                kwargs['reward_model'] = DummyRewardModel(hidden_size)
        
        super().__init__(*args, **kwargs)
    
    def _compute_custom_rewards(self, queries, postprocessed_query_responses, context_length):
        """
        Compute rewards using the custom reward function.
        
        Args:
            queries: Tensor of query token ids [batch, seq]
            postprocessed_query_responses: Full sequence (query + response) [batch, seq]
            context_length: Length of the query/context portion
            
        Returns:
            Tensor of scores for each sequence [batch]
        """
        device = queries.device
        batch_size = queries.shape[0]
        
        # Decode prompts and completions
        prompts = self.processing_class.batch_decode(queries, skip_special_tokens=True)
        completions = self.processing_class.batch_decode(
            postprocessed_query_responses[:, context_length:], 
            skip_special_tokens=True
        )
        
        # Call custom reward function (GRPO-style interface)
        try:
            rewards = self.custom_reward_func(prompts=prompts, completions=completions)
        except TypeError:
            # Simple interface: just completions
            rewards = self.custom_reward_func(completions)
        
        # Convert to tensor
        scores = torch.tensor(rewards, device=device, dtype=torch.float32)
        
        return scores
    
    def _generate_completions_and_compute_rewards(
        self, 
        queries, 
        device, 
        generation_config,
        model,
        ref_policy,
    ):
        """
        Generate completions and compute rewards.
        
        This mirrors the logic in the parent's train() method but uses custom rewards.
        Returns all the tensors needed for PPO training.
        """
        args = self.args
        processing_class = self.processing_class
        accelerator = self.accelerator
        context_length = queries.shape[1]
        
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        sequence_lengths = []
        values = []
        
        # Unwrap model for generation
        from trl.models.utils import unwrap_model_for_generation
        
        generation_kwargs = {"pad_token_id": processing_class.pad_token_id}
        
        with (
            unwrap_model_for_generation(
                model,
                accelerator,
                gather_deepspeed3_params=args.ds3_gather_for_generation,
                generation_kwargs=generation_kwargs,
            ) as unwrapped_model
        ):
            query_responses, logitss = batch_generation(
                unwrapped_model.policy,
                queries,
                args.local_rollout_forward_batch_size,
                processing_class.pad_token_id,
                generation_config,
            )
        
        # Process in batches
        for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
            query = queries[i : i + args.local_rollout_forward_batch_size]
            query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
            response = query_response[:, context_length:]
            logits = logitss[i : i + args.local_rollout_forward_batch_size]
            logprob = selective_log_softmax(logits, response)
            del logits
            empty_cache()
            
            # Compute reference logprobs
            from trl.experimental.ppo.ppo_trainer import forward
            
            if ref_policy is None:
                with self.null_ref_context():
                    ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
            else:
                ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.temperature + 1e-7
            ref_logprob = selective_log_softmax(ref_logits, response)
            del ref_output, ref_logits
            empty_cache()
            
            # Truncate response after stop token
            postprocessed_response = response
            if self.stop_token_id is not None:
                from trl.experimental.ppo.ppo_trainer import truncate_response
                postprocessed_response = truncate_response(
                    self.stop_token_id, processing_class.pad_token_id, response
                )
            
            # Compute sequence lengths
            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
            
            # Compute value estimates
            unwrapped_value_model = accelerator.unwrap_model(model).value_model
            full_value, _, _ = get_reward(
                unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
            )
            value = full_value[:, context_length - 1 : -1].squeeze(-1)
            
            # CUSTOM REWARDS: Use our reward function instead of get_reward()
            if self.custom_reward_func is not None:
                score = self._compute_custom_rewards(query, postprocessed_query_response, context_length)
            else:
                # Fall back to neural reward model
                _, score, _ = get_reward(
                    self.reward_model, postprocessed_query_response, 
                    processing_class.pad_token_id, context_length
                )
            
            responses.append(response)
            postprocessed_responses.append(postprocessed_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)
            scores.append(score)
            values.append(value)
        
        # Concatenate all batches
        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        scores = torch.cat(scores, 0)
        values = torch.cat(values, 0)
        
        return {
            'responses': responses,
            'postprocessed_responses': postprocessed_responses,
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'sequence_lengths': sequence_lengths,
            'scores': scores,
            'values': values,
            'query_responses': query_responses,
            'context_length': context_length,
        }


def main(config_path: str):
    """Main training function."""
    config = load_config(config_path, "grpo")  # Reuse GRPO config structure

    if torch.cuda.is_available():
        print("✓ Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Silicon)")
    else:
        print("⚠ Using CPU")

    print(f"Reward Function: {config.reward_fn}")
    print(f"Model: {config.model_name}")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        name=config.experiment_name or "ppo_experiment",
        algorithm="ppo",
        output_dir=config.output_dir,
        config=asdict(config),
    )
    print(f"Run ID: {tracker.run_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load policy model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Apply LoRA if configured
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
        print("✓ LoRA applied")

    # Create custom reward function (matching GRPO interface)
    reward_fn = make_reward_fn(config.reward_fn)
    
    # Save reward function source code
    verifier = get_verifier(config.reward_fn)
    tracker.save_reward_fn(verifier, name=config.reward_fn)

    # Load reference model
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)

    # Configure PPO
    ppo_config = PPOConfig(
        output_dir=tracker.run_dir,
        total_episodes=config.max_steps if config.max_steps > 0 else 1000,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy if eval_dataset is not None else "no",
        eval_steps=config.eval_steps,
        temperature=config.temperature,
        response_length=config.max_new_tokens,
        num_ppo_epochs=4,
        kl_coef=config.beta,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Initialize trainer with custom reward function
    trainer = CustomRewardPPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_func=reward_fn,  # Our custom reward function!
        value_model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[tracker.get_callback()],
    )

    print(f"\n{'='*50}")
    print("Starting PPO Training with Custom Rewards")
    print(f"{'='*50}\n")

    trainer.train()
    
    trainer.save_model(tracker.run_dir)
    tokenizer.save_pretrained(tracker.run_dir)
    
    print(f"\n✓ Training complete! Model saved to: {tracker.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training with Custom Reward Functions")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
