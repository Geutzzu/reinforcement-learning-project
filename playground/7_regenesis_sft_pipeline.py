"""
ReGenesis-style SFT Dataset Generation Pipeline for Maze Task
==============================================================

This pipeline implements the ReGenesis methodology from:
"ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement"
(arXiv:2410.02108)

The pipeline progresses from ABSTRACT → CONCRETE in 4 steps:
1. Generate maze puzzles
2. Guidance Adaptation: Adapt general reasoning guidelines to maze-specific ones
3. Reasoning Structure Generation: Create reasoning frameworks (without solving)
4. Reasoning Path Generation: Generate full reasoning paths with solutions
5. Filter: Keep only correct solutions
"""

# IMPORTANT: Set multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before any other imports that might initialize CUDA
import multiprocessing
    

import sys
import json
import pandas as pd
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/rl")

from utils.vllm_utils import LLM, VLLM

from main.enigmata import generate_puzzles

# Token counting helper
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    return _tokenizer

def print_token_stats(texts: list[str], stage_name: str):
    """Print avg/min/max token length stats for a list of texts"""
    tokenizer = get_tokenizer()
    tokens = tokenizer(texts)['input_ids']
    lengths = [len(t) for t in tokens]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"  📊 {stage_name} token stats: avg={avg_len:.1f}, min={min_len}, max={max_len}")

# =============================================================================
# 25 SEED PROMPTS FROM REGENESIS PAPER (Table 8)
# =============================================================================
# These are general, task-agnostic reasoning guidelines

SEED_PROMPTS = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this solution?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available.",
    "Try creative thinking, currentGuess and explore unconventional solutions to the problem.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements.",
    "Use Risk Analysis: Evaluate potential risks, weigh their likelihood and impact, and develop strategies to mitigate them.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem?",
    "Stakeholder Analysis: Consider the needs and perspectives of all stakeholders involved.",
    "What is my intuition telling me about the best approach?",
    "What is the best way to test or validate potential solutions?",
    "How can I use feedback and iteration to improve the solution?",
    "What resources (time, money, expertise) are available to address this problem?",
    "Step-by-Step Analysis: Break down the problem into smaller steps and analyze each step in detail.",
    "How can I handle the complexity of this problem?",
]

# =============================================================================
# PROMPTS FOR EACH PIPELINE STEP
# =============================================================================

GUIDANCE_ADAPTATION_PROMPT = """You are adapting a general reasoning guideline to be specific for solving maze navigation problems.

The maze task involves:
- Finding a path from Start (S) at position (1,1) to End (E) at position (height, width)
- Navigating through open cells (.) while avoiding blocked cells (B)
- Moving only up, down, left, or right (no diagonals)
- If no path exists, the answer should be "not exist"

General Reasoning Guideline:
"{seed_prompt}"

Maze Problem:
{maze_problem}

Adapt this general guideline to be specific for solving this maze problem. 
Make it actionable and directly applicable to maze navigation.

Adapted Guideline for Maze:"""

REASONING_STRUCTURE_PROMPT = """You are creating a reasoning structure (framework) for solving a maze navigation problem.
DO NOT solve the maze yet - just create the step-by-step structure that will guide the solution.

Maze Problem:
{maze_problem}

Adapted Reasoning Guideline:
{adapted_guidance}

Create a detailed reasoning structure with numbered steps that outlines HOW to solve this maze.
Focus on the approach, not the actual solution. For example:
1. Identify the start and end positions
2. Analyze the maze layout for possible paths
3. Check for obstacles blocking direct routes
4. etc.

Reasoning Structure:"""

REASONING_PATH_PROMPT = """You are solving a maze navigation problem step by step.

Maze Problem:
{maze_problem}

Follow this reasoning structure:
{reasoning_structure}

Now solve the maze by following the structure above. Show your complete reasoning process, then provide the final answer.

If a path exists, format as: (1,1)->(row,col)->...->(height,width)
If no path exists, answer: not exist

Your answer MUST be wrapped in the following format:
---start_reasoning---
[Your detailed reasoning here]
---end_reasoning---
---start_answer---
[Your path or "not exist"]
---end_answer---

Solution:"""

REASONING_PATH_WITH_HINT_PROMPT = """You are solving a maze navigation problem step by step.

Maze Problem:
{maze_problem}

Follow this reasoning structure:
{reasoning_structure}

HINT: The correct answer is: {hint}

Using this hint, generate a complete reasoning process that arrives at this answer.

Your answer MUST be wrapped in the following format:
---start_reasoning---
[Your detailed reasoning here]
---end_reasoning---
---start_answer---
[Your path or "not exist"]
---end_answer---

Solution:"""


# =============================================================================
# PIPELINE IMPLEMENTATION
# =============================================================================

class ReGenesisDatasetPipeline:
    """
    ReGenesis-style dataset generation pipeline for maze reasoning.
    
    Steps:
    1. Generate puzzles
    2. For each puzzle, for each seed prompt:
       a. Adapt the seed prompt to be maze-specific
       b. Generate a reasoning structure
       c. Generate a reasoning path with solution
    3. Filter to keep only correct solutions
    """
    
    def __init__(self, model_callable, verifier_callable):
        """
        Args:
            model_callable: Function that takes a LIST of prompts and returns LIST of outputs
                           Signature: (prompts: List[str]) -> List[str]
                           Compatible with LLM.predict_batch()
            verifier_callable: Function that verifies if answer is correct
                              Signature: (pred: str, answer: str, meta: dict) -> float
        """
        self.model = model_callable
        self.verifier = verifier_callable
        
    def step1_generate_puzzles(self, task_name: str, count: int, difficulty: str) -> pd.DataFrame:
        """Step 1: Generate maze puzzles"""
        print(f"Step 1: Generating {count} {difficulty} puzzles...")
        df = generate_puzzles(task_name, count=count, difficulty=difficulty)
        print(f"  Generated {len(df)} puzzles")
        return df
    
    def _batch_generate(self, prompts: list[str]) -> list[str]:
        """Call model on a batch of prompts"""
        return self.model(prompts)
    
    def generate_dataset(
        self,
        task_name: str = "maze_2",
        puzzle_count: int = 1000,
        difficulty: str = "hard",
        samples_per_puzzle: int = 5,
        use_hints_for_failures: bool = True,
        min_reward_threshold: float = 1.5,
        batch_size: int = 32,
        debug: bool = False
    ):
        """
        Run the full ReGenesis pipeline with batched inference.
        
        Args:
            task_name: Name of the puzzle task
            puzzle_count: Number of puzzles to generate
            difficulty: Puzzle difficulty
            samples_per_puzzle: Number of different reasoning paths per puzzle
            use_hints_for_failures: Whether to regenerate with hints when initial attempt fails
            min_reward_threshold: Minimum reward to consider a solution correct
            batch_size: Batch size for model inference
            debug: If True, returns (results_df, debug_df) with ALL attempts including failures
            
        Returns:
            If debug=False: DataFrame with successful solutions only
            If debug=True: Tuple of (results_df, debug_df) where debug_df contains ALL attempts
        """
        import random
        
        # Step 1: Generate puzzles
        puzzles_df = self.step1_generate_puzzles(task_name, puzzle_count, difficulty)
        
        # Prepare all puzzle-seed combinations
        print("Step 2: Preparing puzzle-seed combinations...")
        work_items = []
        for idx, row in puzzles_df.iterrows():
            selected_seeds = random.sample(range(len(SEED_PROMPTS)), min(samples_per_puzzle, len(SEED_PROMPTS)))
            for seed_idx in selected_seeds:
                work_items.append({
                    'puzzle_idx': idx,
                    'maze_prompt': row['prompt'],
                    'answer': row['answer'],
                    'meta': row['meta'],
                    'seed_idx': seed_idx,
                    'seed_prompt': SEED_PROMPTS[seed_idx]
                })
        
        print(f"  Total work items: {len(work_items)}")

        # Step 2: Batch adapt guidance
        print("Step 3: Adapting guidance (batched)...")
        print(f" Example Guidance Adaptation Prompt:\n{GUIDANCE_ADAPTATION_PROMPT.format(
            seed_prompt=work_items[0]['seed_prompt'],
            maze_problem=work_items[0]['maze_prompt']
        )}")

        adapt_prompts = [
            GUIDANCE_ADAPTATION_PROMPT.format(
                seed_prompt=item['seed_prompt'],
                maze_problem=item['maze_prompt']
            )
            for item in work_items
        ]
        
        adapted_guidances = []
        for i in tqdm(range(0, len(adapt_prompts), batch_size), desc="Adapting guidance"):
            batch = adapt_prompts[i:i+batch_size]
            adapted_guidances.extend(self._batch_generate(batch))
        print_token_stats(adapted_guidances, "Adapted Guidances")
        
        # Step 3: Batch generate structures
        print("Step 4: Generating reasoning structures (batched)...")
        print(f" Example Reasoning Structure Prompt:\n{REASONING_STRUCTURE_PROMPT.format(
            maze_problem=work_items[0]['maze_prompt'],
            adapted_guidance=adapted_guidances[0]
        )}")

        structure_prompts = [
            REASONING_STRUCTURE_PROMPT.format(
                maze_problem=work_items[i]['maze_prompt'],
                adapted_guidance=adapted_guidances[i]
            )
            for i in range(len(work_items))
        ]
        
        reasoning_structures = []
        for i in tqdm(range(0, len(structure_prompts), batch_size), desc="Generating structures"):
            batch = structure_prompts[i:i+batch_size]
            reasoning_structures.extend(self._batch_generate(batch))
        print_token_stats(reasoning_structures, "Reasoning Structures")
        
        # Step 4: Batch generate reasoning paths
        print("Step 5: Generating reasoning paths (batched)...")
        print(f" Example Reasoning Path Prompt:\n{REASONING_PATH_PROMPT.format(
            maze_problem=work_items[0]['maze_prompt'],
            reasoning_structure=reasoning_structures[0]
        )}")

        path_prompts = [
            REASONING_PATH_PROMPT.format(
                maze_problem=work_items[i]['maze_prompt'],
                reasoning_structure=reasoning_structures[i]
            )
            for i in range(len(work_items))
        ]
        
        reasoning_paths = []
        for i in tqdm(range(0, len(path_prompts), batch_size), desc="Generating paths"):
            batch = path_prompts[i:i+batch_size]
            reasoning_paths.extend(self._batch_generate(batch))
        print_token_stats(reasoning_paths, "Reasoning Paths")
        
        # Step 5: Verify and filter
        print("Step 6: Verifying solutions...")
        results = []
        failed_items = []

        print(f" Example Reasoning Path:\n{reasoning_paths[0]}\n")
        
        for i, item in enumerate(tqdm(work_items, desc="Verifying")):
            reasoning_path = reasoning_paths[i]
            reward = self.verifier(reasoning_path, item['answer'], item['meta'])
            
            if reward >= min_reward_threshold:
                results.append({
                    'prompt': item['maze_prompt'],
                    'answer': item['answer'],
                    'meta': item['meta'],
                    'reasoning_path': reasoning_path,
                    'adapted_guidance': adapted_guidances[i],
                    'reasoning_structure': reasoning_structures[i],
                    'seed_prompt_idx': item['seed_idx'],
                    'seed_prompt': item['seed_prompt'],
                    'reward': reward
                })
            elif use_hints_for_failures:
                failed_items.append((i, item, reasoning_structures[i]))
        
        # Step 6: Retry failed items with hints
        if use_hints_for_failures and failed_items:
            print(f"Step 7: Retrying {len(failed_items)} failed items with hints...")
            hint_prompts = [
                REASONING_PATH_WITH_HINT_PROMPT.format(
                    maze_problem=item['maze_prompt'],
                    reasoning_structure=structure,
                    hint=item['answer']
                )
                for _, item, structure in failed_items
            ]
            
            hint_paths = []
            for i in tqdm(range(0, len(hint_prompts), batch_size), desc="Retrying with hints"):
                batch = hint_prompts[i:i+batch_size]
                hint_paths.extend(self._batch_generate(batch))
            
            for j, (orig_idx, item, structure) in enumerate(failed_items):
                reasoning_path = hint_paths[j]
                reward = self.verifier(reasoning_path, item['answer'], item['meta'])
                
                if reward >= min_reward_threshold:
                    results.append({
                        'prompt': item['maze_prompt'],
                        'answer': item['answer'],
                        'meta': item['meta'],
                        'reasoning_path': reasoning_path,
                        'adapted_guidance': adapted_guidances[orig_idx],
                        'reasoning_structure': structure,
                        'seed_prompt_idx': item['seed_idx'],
                        'seed_prompt': item['seed_prompt'],
                        'reward': reward,
                        'used_hint': True
                    })

            print(f" Example Reasoning Path with Hint:\n{hint_paths[0]}\n")

        # Build debug dataframe with ALL attempts
        if debug:
            debug_records = []
            for i, item in enumerate(work_items):
                # Check if this item succeeded on first try
                first_try_success = any(
                    r['prompt'] == item['maze_prompt'] and 
                    r['seed_prompt_idx'] == item['seed_idx'] and 
                    not r.get('used_hint', False)
                    for r in results
                )
                # Check if it succeeded with hint
                hint_success = any(
                    r['prompt'] == item['maze_prompt'] and 
                    r['seed_prompt_idx'] == item['seed_idx'] and 
                    r.get('used_hint', False)
                    for r in results
                )
                
                # Compute reward for first attempt
                first_reward = self.verifier(reasoning_paths[i], item['answer'], item['meta'])
                
                debug_records.append({
                    'puzzle_idx': item['puzzle_idx'],
                    'seed_idx': item['seed_idx'],
                    'seed_prompt': item['seed_prompt'],
                    'maze_prompt': item['maze_prompt'],
                    'answer': item['answer'],
                    'meta': item['meta'],
                    'adapted_guidance': adapted_guidances[i],
                    'reasoning_structure': reasoning_structures[i],
                    'reasoning_path': reasoning_paths[i],
                    'reward': first_reward,
                    'success': first_try_success or hint_success,
                    'first_try_success': first_try_success,
                    'hint_success': hint_success,
                })
            
            debug_df = pd.DataFrame(debug_records)
        
        result_df = pd.DataFrame(results)
        print(f"\nPipeline complete!")
        print(f"  Input puzzles: {len(puzzles_df)}")
        print(f"  Samples per puzzle: {samples_per_puzzle}")
        print(f"  Total attempts: {len(work_items)}")
        print(f"  Correct solutions: {len(result_df)}")
        print(f"  Success rate: {len(result_df) / len(work_items) * 100:.1f}%")
        
        if debug:
            return result_df, debug_df
        return result_df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from Enigmata.verifiable_tasks.tasks.maze_2.verifier import verify

    llm = VLLM(
        model="Qwen/Qwen3-4B-Instruct-2507", 
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )

    def model_fn(prompts: list[str]) -> list[str]:
        return llm.predict_batch(
            prompts,
            max_output_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0
        )

    pipeline = ReGenesisDatasetPipeline(
        model_callable=model_fn,
        verifier_callable=verify
    )

    df, debug_df = pipeline.generate_dataset(
        task_name='maze_2',
        puzzle_count=100,
        difficulty='medium',
        samples_per_puzzle=3,
        batch_size=20000,
        debug=True
    )

    df.to_parquet('maze_sft_regenesis.parquet')
    debug_df.to_parquet('maze_sft_regenesis_debug.parquet')





