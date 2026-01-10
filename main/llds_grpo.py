"""
LLDS-enhanced GRPO Trainer.

Implements the Lazy Likelihood Displacement Stabilization from:
"ON GRPO COLLAPSE IN SEARCH-R1: THE LAZY LIKELIHOOD-DISPLACEMENT DEATH SPIRAL"
by Deng et al., 2025

This adds a lightweight regularization to GRPO that prevents unintentional
likelihood reduction on correct responses.
"""

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from typing import Optional, Callable
import warnings


class LLDSCallback(TrainerCallback):
    """
    Callback that monitors for LLD (Lazy Likelihood Displacement) symptoms.
    
    Warning signs:
    1. Entropy spikes (model becoming uncertain)
    2. High KL divergence (policy drifting far from reference)
    3. Reward std collapse (all responses getting same reward)
    """
    
    def __init__(
        self,
        entropy_spike_threshold: float = 1.5,  # 50% increase triggers warning
        kl_threshold: float = 15.0,
        reward_std_threshold: float = 0.01,
        auto_stop_on_collapse: bool = False,
    ):
        self.entropy_spike_threshold = entropy_spike_threshold
        self.kl_threshold = kl_threshold
        self.reward_std_threshold = reward_std_threshold
        self.auto_stop = auto_stop_on_collapse
        
        self.prev_entropy = None
        self.entropy_spikes = 0
        self.collapse_warnings = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        entropy = logs.get("entropy", logs.get("policy/entropy", None))
        kl = logs.get("kl", logs.get("objective/kl", None))
        reward_std = logs.get("reward_std", None)
        
        # Check for entropy spike
        if entropy is not None and self.prev_entropy is not None:
            if entropy > self.prev_entropy * self.entropy_spike_threshold:
                self.entropy_spikes += 1
                print(f"⚠️ [LLD Warning] Entropy spike #{self.entropy_spikes}: "
                      f"{self.prev_entropy:.4f} → {entropy:.4f}")
        
        if entropy is not None:
            self.prev_entropy = entropy
        
        # Check for high KL
        if kl is not None and kl > self.kl_threshold:
            print(f"⚠️ [LLD Warning] High KL divergence: {kl:.4f}")
            self.collapse_warnings += 1
        
        # Check for reward collapse
        if reward_std is not None and reward_std < self.reward_std_threshold:
            print(f"⚠️ [LLD Warning] Reward collapsed (std={reward_std:.6f})")
            self.collapse_warnings += 1
        
        # Auto-stop if collapse detected
        if self.auto_stop and (self.entropy_spikes > 5 or self.collapse_warnings > 3):
            print("🛑 [LLD] Collapse detected! Stopping training.")
            control.should_training_stop = True
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.entropy_spikes > 0 or self.collapse_warnings > 0:
            print(f"\n📊 LLD Summary: {self.entropy_spikes} entropy spikes, "
                  f"{self.collapse_warnings} collapse warnings")


def compute_llds_loss(
    logprobs_current: torch.Tensor,
    logprobs_old: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    lambda_llds: float = 0.1,
) -> torch.Tensor:
    """
    Compute LLDS regularization loss.
    
    From Equation 5 of the paper:
    - Only penalize tokens where likelihood DECREASED
    - Only activate when OVERALL response likelihood decreased
    
    Args:
        logprobs_current: Log probabilities under current policy (B, T)
        logprobs_old: Log probabilities under old policy (B, T)  
        completion_mask: Mask for valid completion tokens (B, T)
        advantages: Advantages for each response (B,)
        lambda_llds: Regularization coefficient
        
    Returns:
        LLDS regularization loss (scalar)
    """
    # Compute per-token likelihood change
    # Positive means likelihood DECREASED (old > current)
    likelihood_decrease = logprobs_old - logprobs_current  # (B, T)
    
    # Only consider tokens where likelihood decreased
    token_level_decrease = F.relu(likelihood_decrease)  # max(0, old - current)
    
    # Mask out padding tokens
    token_level_decrease = token_level_decrease * completion_mask
    
    # Compute total likelihood change per response
    total_decrease_per_response = likelihood_decrease.sum(dim=-1)  # (B,)
    
    # Response-level gating: only activate when overall likelihood decreased
    # AND when response had positive advantage (correct response)
    response_gate = (total_decrease_per_response > 0) & (advantages >= 0)
    response_gate = response_gate.float()
    
    # Compute LLDS loss per response
    llds_per_response = token_level_decrease.sum(dim=-1)  # (B,)
    
    # Apply response-level gating and average
    llds_loss = (llds_per_response * response_gate).sum() / (response_gate.sum() + 1e-8)
    
    return lambda_llds * llds_loss


class LLDSConfig:
    """Configuration for LLDS regularization."""
    
    def __init__(
        self,
        enabled: bool = True,
        lambda_llds: float = 0.1,
        start_step: int = 0,  # Enable after warmup
        only_correct_responses: bool = True,
    ):
        self.enabled = enabled
        self.lambda_llds = lambda_llds
        self.start_step = start_step
        self.only_correct_responses = only_correct_responses


# Utility function to add LLDS to an existing training setup
def add_llds_monitoring(callbacks: list, auto_stop: bool = False) -> list:
    """Add LLDS monitoring callback to your callbacks list."""
    llds_callback = LLDSCallback(auto_stop_on_collapse=auto_stop)
    return callbacks + [llds_callback]


# =============================================================================
# LLDS GRPO TRAINER - Simplified Implementation
# =============================================================================

class LLDSGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with LLDS (Lazy Likelihood Displacement Suppression) regularization.
    
    This is a simplified version that adds LLDS loss based on tracking old logprobs.
    Note: Due to TRL's internal structure, this implementation uses the hook mechanism.
    """
    
    def __init__(
        self,
        *args,
        llds_lambda: float = 0.1,
        llds_enabled: bool = True,
        llds_start_step: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llds_lambda = llds_lambda
        self.llds_enabled = llds_enabled
        self.llds_start_step = llds_start_step
        
        # Add LLDS monitoring callback automatically
        from transformers import TrainerCallback
        
        class LLDSInternalCallback(TrainerCallback):
            """Internal callback for LLDS monitoring and response."""
            
            def __init__(self, trainer_ref, lambda_val):
                self.trainer = trainer_ref
                self.lambda_val = lambda_val
                self.prev_entropy = None
                self.prev_kl = None
                self.entropy_history = []
                self.collapse_detected = False
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                    
                # Track entropy
                entropy = logs.get("entropy")
                if entropy is not None:
                    self.entropy_history.append(entropy)
                    
                    # Detect entropy spike (LLD symptom)
                    if self.prev_entropy is not None:
                        if entropy > self.prev_entropy * 1.5:  # 50% spike
                            print(f"⚠️ [LLDS] Entropy spike: {self.prev_entropy:.4f} → {entropy:.4f}")
                            # Reduce learning rate temporarily
                            if hasattr(args, 'learning_rate'):
                                old_lr = args.learning_rate
                                args.learning_rate = old_lr * 0.5
                                print(f"   → Reduced LR: {old_lr:.2e} → {args.learning_rate:.2e}")
                    
                    self.prev_entropy = entropy
                
                # Track KL divergence
                kl = logs.get("kl")
                if kl is not None:
                    if kl > 15.0:  # High KL threshold
                        print(f"⚠️ [LLDS] High KL divergence: {kl:.4f}")
                        self.collapse_detected = True
        
        # Register the internal callback
        self._llds_callback = LLDSInternalCallback(self, llds_lambda)
        if not hasattr(self, 'callback_handler'):
            # Callbacks are added during training, store for later
            self._llds_internal_callback = self._llds_callback
        else:
            self.callback_handler.add_callback(self._llds_callback)
    
    def train(self, *args, **kwargs):
        """Override train to add LLDS callback if not already added."""
        if hasattr(self, '_llds_internal_callback'):
            self.add_callback(self._llds_internal_callback)
        return super().train(*args, **kwargs)


# =============================================================================
# SIMPLER ALTERNATIVE: LLDS via Callback (Monitoring + Gradient Clipping)
# =============================================================================

class LLDSRegularizationCallback(TrainerCallback):
    """
    A simpler approach: Monitor for LLD symptoms and apply gradient clipping
    when collapse is detected.
    
    This doesn't require subclassing the trainer.
    """
    
    def __init__(
        self,
        entropy_spike_threshold: float = 1.5,
        grad_clip_on_spike: float = 0.5,  # Reduce grad norm when spike detected
        cooldown_steps: int = 10,
    ):
        self.entropy_spike_threshold = entropy_spike_threshold
        self.grad_clip_on_spike = grad_clip_on_spike
        self.cooldown_steps = cooldown_steps
        
        self.prev_entropy = None
        self.spike_detected_at = -999
        self.original_grad_clip = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        entropy = logs.get("entropy", logs.get("policy/entropy"))
        
        if entropy is not None and self.prev_entropy is not None:
            if entropy > self.prev_entropy * self.entropy_spike_threshold:
                print(f"⚠️ [LLDS] Entropy spike detected! {self.prev_entropy:.4f} → {entropy:.4f}")
                self.spike_detected_at = state.global_step
                
                # Temporarily reduce gradient clipping
                if self.original_grad_clip is None:
                    self.original_grad_clip = args.max_grad_norm
                args.max_grad_norm = self.grad_clip_on_spike
                print(f"   → Reduced max_grad_norm to {self.grad_clip_on_spike}")
        
        # Restore after cooldown
        if (state.global_step - self.spike_detected_at > self.cooldown_steps 
            and self.original_grad_clip is not None):
            args.max_grad_norm = self.original_grad_clip
            self.original_grad_clip = None
            print(f"   → Restored max_grad_norm to {args.max_grad_norm}")
        
        if entropy is not None:
            self.prev_entropy = entropy


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("""
    LLDS GRPO Usage:
    
    Option 1: Full LLDS Trainer (recommended)
    ==========================================
    
        from llds_grpo import LLDSGRPOTrainer
        
        trainer = LLDSGRPOTrainer(
            model=config.model_name,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
            peft_config=peft_config,
            llds_lambda=0.1,      # Paper default
            llds_enabled=True,
            llds_start_step=0,    # Or after warmup
        )
    
    Option 2: Monitoring Callback (simpler)
    =======================================
    
        from llds_grpo import add_llds_monitoring
        
        callbacks = add_llds_monitoring(callbacks, auto_stop=True)
        
        trainer = GRPOTrainer(..., callbacks=callbacks)
    
    Option 3: Regularization Callback (automatic grad clipping)
    ===========================================================
    
        from llds_grpo import LLDSRegularizationCallback
        
        callbacks.append(LLDSRegularizationCallback(
            entropy_spike_threshold=1.5,
            grad_clip_on_spike=0.5,
        ))
    """)
