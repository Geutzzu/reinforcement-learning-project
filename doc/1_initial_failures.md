# GRPO Maze Experiments: Failure Analysis

## Overview

This document summarizes the key failures encountered during initial GRPO (Group Relative Policy Optimization) experiments on the maze pathfinding task. Each experiment revealed different failure modes that are common in reinforcement learning for language models.

---

## Experiment 1: Base Model + Sparse Rewards

### Configuration
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (base, no SFT)
- **Reward**: Binary (1.0 for correct path, 0.0 otherwise)
- **num_generations**: 4-8

### Failure Mode: Reward Hacking via Format Exploitation

The base model naturally outputs Python code (as it's trained for code generation). The verifier was designed to extract coordinates from code blocks using regex:

```python
coordinates = re.findall(r'\(\d+,\s*\d+\)', normalized_pred)
```

**Problem**: This regex matches ANY coordinate-like pattern, including:
- Direction vectors: `(0, 1), (1, 0)` 
- Variable assignments: `start = (1, 1)`, `end = (5, 5)`
- Random coordinates in code: `queue.append((3, 4))`

**Result**: The model received reward 1.0 when Python code **accidentally** contained coordinates that formed a valid path sequence. The model learned to generate code patterns that maximized the probability of incidental coordinate matches, rather than actually solving mazes.

### Key Metrics Observed
- Reward oscillating between 0.4-0.6 (random chance)
- Model generating verbose Python code (~400 tokens)
- No convergence toward the expected output format

### Lesson Learned
Sparse binary rewards on base models with loose format validation leads to reward hacking. The model exploits the easiest path to reward, not the intended solution.

---

## Experiment 2: SFT Model + 4 Generations

### Configuration
- **Model**: SFT-trained on maze task (outputs path format)
- **Reward**: Binary (1.0 for correct path, 0.0 otherwise)
- **num_generations**: 4
- **temperature**: 0.7

### Failure Mode: Rapid Mode Collapse to "Not Exist"

The SFT model was already trained to output the correct format:
```
(1,1)->(2,1)->(3,1)->(4,1)->(5,5)
```

However, it quickly collapsed to always outputting:
```
The answer is not exist the path from start to end.
```

**Why this happened**:
1. The dataset contained ~10-20% unsolvable mazes
2. For those mazes, "not exist" is the correct answer (reward 1.0)
3. The model discovered this "safe" output that always gets reward ≥0.1
4. With only 4 generations, there wasn't enough diversity to escape

**Key Metrics at Collapse**:
| Metric | Value |
|--------|-------|
| `loss` | 0.0 |
| `grad_norm` | 0.0 |
| `frac_reward_zero_std` | 1.0 |
| `reward` | ~0.15 (matching unsolvable maze frequency) |

### Lesson Learned
With few generations and a confident SFT model, the policy can quickly converge to a single low-risk output. The lack of diversity (all 4 generations identical) means zero advantage signal, causing training to stall.

---

## Experiment 3: SFT Model + 16 Generations

### Configuration
- **Model**: SFT-trained on maze task
- **Reward**: Binary (1.0 for correct path, 0.0 otherwise)
- **num_generations**: 16
- **temperature**: 0.7
- **batch_size**: 16

### Failure Mode: Delayed Collapse to Short Identical Outputs

With 16 generations, training initially showed promise:
- Loss was non-zero (~0.12)
- Gradients were flowing (grad_norm ~2.3)
- Some reward variance existed (reward_std ~0.1)

However, after ~100-150 steps, collapse occurred:

| Step | loss | grad_norm | frac_reward_zero_std | completion_length |
|------|------|-----------|----------------------|-------------------|
| 110 | 0.12 | 2.3 | 0.60 | 15.8 |
| 130 | 0.03 | 0.0 | 0.95 | 14.1 |
| 150 | 0.03 | 0.0 | 0.95 | 14.0 |

**What happened**:
1. The model learned to output very short, high-confidence responses (~14 tokens)
2. Temperature 0.7 wasn't enough diversity for 16 identical short outputs
3. Binary rewards meant partial progress wasn't rewarded
4. Once all 16 generations were identical, advantage = 0 for all

### Lesson Learned
More generations delay collapse but don't prevent it with sparse rewards. The SFT model's confidence creates an "attractor" toward a single output mode. Binary rewards provide no gradient when all generations get the same score.

---

## Root Cause Summary

| Experiment | Primary Failure | Root Cause |
|------------|-----------------|------------|
| Base + Sparse | Reward hacking | Loose format validation + base model outputs code |
| SFT + 4 gen | Fast collapse | Low diversity + confident model + sparse rewards |
| SFT + 16 gen | Delayed collapse | Same issues, just slower convergence |

---

## Recommended Fixes

### 1. Partial/Dense Rewards
Instead of binary 0/1, provide gradient signal for partial correctness:
- +0.1 for correct format markers
- +0.2 for starting at (1,1)
- +0.3 for valid moves
- +0.4 for reaching end

### 2. Higher Temperature
Increase from 0.7 to 1.0+ to maintain diversity in generations.

### 3. KL Penalty (beta > 0)
Add regularization to prevent policy from drifting too far:
```yaml
beta: 0.01
```

### 4. Strict Format Validation
Use format-specific markers that can't be accidentally matched:
```
---start_answer---
(1,1)->(2,1)->(5,5)
---end_answer---
```

### 5. Lower Learning Rate
Reduce from 1e-5 to 5e-6 for more stable updates.

---

## Conclusion

These experiments demonstrate that GRPO with sparse rewards on a confident policy (SFT model) is prone to mode collapse. The combination of:
- Binary rewards (no partial credit)
- High model confidence (low output diversity)
- Short horizon before collapse (~100-150 steps)

...creates a fragile training dynamic. Successful GRPO training requires dense reward signals and mechanisms to maintain policy diversity.
