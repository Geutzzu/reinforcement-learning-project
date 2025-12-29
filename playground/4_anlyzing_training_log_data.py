"""
Analyze and compare GRPO training experiments
"""
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = "/Users/geo/facultate/rl/rl/results/maze_2_grpo_quick"

def parse_training_log(log_path):
    """Extract metrics from training.log"""
    metrics = {
        'steps': [],
        'rewards': [],
        'reward_std': [],
        'loss': [],
        'lr': [],
        'frac_zero_std': [],
        'grad_norm': [],
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    step = 0
    for line in lines:
        if not line.startswith("{'loss':"):
            continue
        
        try:
            # Parse as Python dict (using eval carefully on known format)
            data = eval(line.strip())
            step += 10  # Assuming logging every 10 steps
            
            metrics['steps'].append(step)
            metrics['loss'].append(data.get('loss', 0))
            metrics['lr'].append(data.get('learning_rate', 0))
            metrics['rewards'].append(data.get('reward', 0))
            metrics['reward_std'].append(data.get('reward_std', 0))
            metrics['frac_zero_std'].append(data.get('frac_reward_zero_std', 0))
            metrics['grad_norm'].append(data.get('grad_norm', 0))
        except:
            continue
    
    return metrics

def parse_config(exp_dir):
    """Try to extract config from experiment directory"""
    config = {}
    
    # Try adapter_config.json for base model
    adapter_config = exp_dir / "adapter_config.json"
    if adapter_config.exists():
        try:
            with open(adapter_config) as f:
                ac = json.load(f)
                base = ac.get('base_model_name_or_path', 'unknown')
                # Extract just the model name
                if '/' in base:
                    config['base_model'] = base.split('/')[-1][:20]
                else:
                    config['base_model'] = base[:20]
        except:
            pass
    
    # Try to extract from log - first entry learning rate
    log_path = exp_dir / "training.log"
    if log_path.exists():
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith("{'loss':"):
                    try:
                        data = eval(line.strip())
                        config['initial_lr'] = data.get('learning_rate', 0)
                        break
                    except:
                        pass
    
    return config

def main():
    experiments = {}
    
    # Collect all experiments
    for exp_dir in Path(RESULTS_DIR).iterdir():
        if not exp_dir.is_dir():
            continue
        
        log_path = exp_dir / "training.log"
        if not log_path.exists():
            continue
        
        exp_name = exp_dir.name
        metrics = parse_training_log(log_path)
        config = parse_config(exp_dir)
        
        if len(metrics['rewards']) > 0:
            experiments[exp_name] = {
                'metrics': metrics,
                'config': config,
                'num_steps': max(metrics['steps']) if metrics['steps'] else 0,
            }
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Print summary table
    print("=" * 120)
    print("EXPERIMENT SUMMARY")
    print("=" * 120)
    print(f"{'Experiment':<55} {'Steps':>7} {'Mean R':>8} {'Final R':>8} {'Max R':>8} {'Init LR':>12}")
    print("-" * 120)
    
    for name, data in sorted(experiments.items()):
        rewards = data['metrics']['rewards']
        config = data['config']
        init_lr = config.get('initial_lr', 0)
        print(f"{name[:55]:<55} {data['num_steps']:>7} {np.mean(rewards):>8.3f} {rewards[-1]:>8.3f} {max(rewards):>8.3f} {init_lr:>12.2e}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Plot 1: Reward over steps
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(sorted(experiments.items())):
        steps = data['metrics']['steps']
        rewards = data['metrics']['rewards']
        short_name = name[:30] + "..." if len(name) > 30 else name
        ax1.plot(steps, rewards, label=short_name, color=colors[i], marker='o', markersize=3, alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward over Training Steps')
    ax1.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate decay
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(sorted(experiments.items())):
        steps = data['metrics']['steps']
        lr = data['metrics']['lr']
        short_name = name[:30] + "..." if len(name) > 30 else name
        ax2.plot(steps, lr, label=short_name, color=colors[i], marker='o', markersize=3, alpha=0.8)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Decay')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fraction with zero std (no learning signal)
    ax3 = axes[1, 0]
    for i, (name, data) in enumerate(sorted(experiments.items())):
        steps = data['metrics']['steps']
        frac = data['metrics']['frac_zero_std']
        short_name = name[:30] + "..." if len(name) > 30 else name
        ax3.plot(steps, frac, label=short_name, color=colors[i], marker='o', markersize=3, alpha=0.8)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Fraction Zero Std')
    ax3.set_title('Batches with No Reward Variance')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics comparison (bar chart)
    ax4 = axes[1, 1]
    names = list(sorted(experiments.keys()))
    short_names = [n[:18] + "..." if len(n) > 18 else n for n in names]
    mean_rewards = [np.mean(experiments[n]['metrics']['rewards']) for n in names]
    max_rewards = [max(experiments[n]['metrics']['rewards']) for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax4.bar(x - width/2, mean_rewards, width, label='Mean Reward', color='steelblue')
    ax4.bar(x + width/2, max_rewards, width, label='Max Reward', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward Comparison Across Experiments')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = '/Users/geo/facultate/rl/rl/playground/grpo_experiments_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    # Don't show - just save
    plt.close()

if __name__ == "__main__":
    main()
