# =============================================================================
# CELL 1: Install dependencies (run once, then restart kernel)
# =============================================================================
# !pip uninstall -y numpy scipy -qq
# !pip install numpy==1.26.4 scipy==1.12.0 -qq  
# !pip install vllm transformers pandas tqdm pillow -qq
# print("✅ Restart kernel now!")
# =============================================================================

# =============================================================================
# CELL 2: Main code starts here (after kernel restart)
# =============================================================================
"""
ReGenesis-style SFT Dataset Generation Pipeline for Maze Task (Kaggle Bundle)
==============================================================================

FULL STANDALONE BUNDLE - Copy this entire file into a Kaggle notebook cell.

This pipeline implements the ReGenesis methodology from:
"ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement"
(arXiv:2410.02108)

REQUIREMENTS:
    !pip install vllm transformers pandas tqdm pillow

GPU RECOMMENDATION: Use "GPU T4 x2" (32GB total) for Qwen3-VL-8B-Instruct

USAGE:
    Run this entire cell, then see the "RUN PIPELINE" section at the bottom.
"""

import re
import json
import random
import collections
import pandas as pd
from typing import List, Optional, Any
from tqdm import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================

KAGGLE_OUTPUT_PATH = "/kaggle/working/"

# =============================================================================
# MAZE PROMPT TEMPLATE
# =============================================================================

MAZE_PROMPT_TEMPLATE = """Given a 5x5 maze map, as shown below:

{question}

Where:
S represents the start point (located in the top-left corner at coordinates (1, 1))
E represents the end point (located in the bottom-right corner at coordinates (5, 5))
B represents an obstacle (impassable)
. represents open space (passable)

Rules:
1. You can only move up, down, left, or right, not diagonally.
2. You cannot pass through obstacles (B).
3. You can move freely on open spaces (.).
4. The goal is to find a path from the start point (S) to the end point (E).

Find a valid path from start to end.

**STRICT OUTPUT FORMAT - DO NOT DEVIATE:**

Your response MUST contain ONLY these two blocks with NO text before, between, or after them.

If path EXISTS:
---start_reasoning---
<brief reasoning - be concise>
---end_reasoning---
---start_answer---
(1,1)->(r,c)->...->(5,5)
---end_answer---

If NO path exists:
---start_reasoning---
<brief explanation why no path>
---end_reasoning---
---start_answer---
not exist
---end_answer---

CRITICAL:
- Keep reasoning SHORT and to the point
- NO tokens outside the two blocks
- Answer block: ONLY the path OR "not exist"
"""

# =============================================================================
# MAZE GENERATOR (from Enigmata/verifiable_tasks/tasks/maze_2/generator.py)
# =============================================================================

def generate_maze(width, height, percentage=0.35):
    maze = [['.' for _ in range(width)] for _ in range(height)]
    
    num_obstacles = int((width * height) * percentage)
    for _ in range(num_obstacles):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        maze[y][x] = 'B'
    
    maze[0][0] = 'S'
    maze[height-1][width-1] = 'E'
    
    return maze


def maze_to_str(maze):
    cols = len(maze[0])
    header = "    " + " ".join(str(i+1) for i in range(cols))
    rows = [f"{i+1}   " + " ".join(row) for i, row in enumerate(maze)]
    return header + "\n" + "\n".join(rows)


def generate_reasoning(maze, path, explored_cells):
    rows, cols = len(maze), len(maze[0])
    
    intros = [
        "I need to find a path from S at (1,1) to E at ({},{}).".format(rows, cols),
        "Starting at (1,1), I'll explore the maze to reach ({},{}).".format(rows, cols),
        "Let me trace a path from the start S to the end E.",
        "I'll navigate from (1,1) to ({},{}) avoiding obstacles.".format(rows, cols),
    ]
    
    reasoning_parts = [random.choice(intros)]
    
    if path is None:
        explored_set = set(explored_cells)
        end_x, end_y = rows - 1, cols - 1
        
        def distance_to_end(cell):
            return abs(cell[0] - end_x) + abs(cell[1] - end_y)
        
        closest_cells = sorted(explored_cells, key=distance_to_end)[:3]
        
        if len(explored_cells) > 1:
            reasoning_parts.append("Exploring from (1,1), I can reach {} cells.".format(len(explored_cells)))
        
        if closest_cells:
            closest = closest_cells[0]
            cx, cy = closest
            dist = distance_to_end(closest)
            
            reasoning_parts.append("Closest reachable cell to E is ({},{}) at distance {}.".format(cx+1, cy+1, dist))
            
            blocked_neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] == 'B':
                        blocked_neighbors.append("({},{})".format(nx+1, ny+1))
            
            if blocked_neighbors:
                reasoning_parts.append("From ({},{}), blocked by obstacles at: {}.".format(
                    cx+1, cy+1, ", ".join(blocked_neighbors)))
        
        conclusions = [
            "No valid path exists - E is cut off by obstacles.",
            "Cannot reach destination - all routes blocked.",
            "The obstacles form a barrier preventing access to E.",
        ]
        reasoning_parts.append(random.choice(conclusions))
    else:
        path_len = len(path)
        
        if path_len <= 6:
            steps_to_describe = list(range(1, path_len))
        else:
            steps_to_describe = [1, 2]
            middle_steps = list(range(3, path_len - 2))
            if middle_steps:
                steps_to_describe.extend(random.sample(middle_steps, min(2, len(middle_steps))))
            steps_to_describe.extend([path_len - 2, path_len - 1])
            steps_to_describe = sorted(set(steps_to_describe))
        
        for idx in steps_to_describe:
            if idx >= path_len:
                continue
            prev = path[idx - 1]
            curr = path[idx]
            
            options = []
            blocked = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = prev[0] + dx, prev[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] == 'B':
                        blocked.append("({},{})".format(nx+1, ny+1))
                    elif maze[nx][ny] != 'B':
                        options.append("({},{})".format(nx+1, ny+1))
            
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            direction = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}.get((dx, dy), "")
            
            if len(options) == 1:
                phrase = "From ({},{}), only option is {} - moving {}.".format(
                    prev[0]+1, prev[1]+1, options[0], direction)
            elif blocked:
                phrase = "At ({},{}), {} blocked. Going {} to ({},{}).".format(
                    prev[0]+1, prev[1]+1, blocked[0], direction, curr[0]+1, curr[1]+1)
            else:
                phrase = "From ({},{}), choosing {} to ({},{}).".format(
                    prev[0]+1, prev[1]+1, direction, curr[0]+1, curr[1]+1)
            
            reasoning_parts.append(phrase)
        
        if path_len > 1:
            last = path[-1]
            reasoning_parts.append("Reached E at ({},{}).".format(last[0]+1, last[1]+1))
        
        conclusions = [
            "Path found with {} steps total.".format(path_len),
            "Successfully navigated in {} moves.".format(path_len - 1),
            "Complete path has {} coordinates.".format(path_len),
        ]
        reasoning_parts.append(random.choice(conclusions))
    
    return "\n".join(reasoning_parts)


def get_maze_answer(maze, lang='en'):
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    explored = []

    def bfs(maze, start, end):
        queue = collections.deque([(start, [start])])
        visited = set([start])

        while queue:
            (x, y), path = queue.popleft()
            explored.append((x, y))
            
            if (x, y) == end:
                return path

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and maze[nx][ny] != 'B':
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))

        return None

    path = bfs(maze, start, end)
    reasoning = generate_reasoning(maze, path, explored)

    if path:
        path_str = "->".join(f"({x+1},{y+1})" for x, y in path)
        answer = f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
{path_str}
---end_answer---"""
    else:
        answer = f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
not exist
---end_answer---"""

    return answer


def generate_puzzles(count=100, difficulty='medium'):
    """Generate maze puzzles"""
    height = 5
    width = 5
    exist = {}
    dif_level = {"easy": [15, 25], "medium": [26, 40], "hard": [41, 55]}
    
    puzzles = []
    for i in tqdm(range(count), desc="Generating puzzles"):
        while True:
            p = random.randint(dif_level[difficulty][0], dif_level[difficulty][1]) / 100
            num_obs = int(p * height * width)
            maze = generate_maze(width, height, p)
            has_str = maze_to_str(maze)
            if has_str in exist:
                continue
            else:
                exist[has_str] = 1
                break
        
        answer = get_maze_answer(maze)
        puzzles.append({
            "prompt": MAZE_PROMPT_TEMPLATE.format(question=maze_to_str(maze)),
            "answer": answer,
            "meta": json.dumps({
                "id": f"maze_2_{difficulty}_{i}",
                "question": maze,
                "width": width,
                "height": height,
                "num_obstacles": num_obs,
                "answer": answer,
                "difficulty_level": difficulty,
            }),
        })
    
    return pd.DataFrame(puzzles)


# =============================================================================
# MAZE VERIFIER (from Enigmata/verifiable_tasks/tasks/maze_2/verifier.py)
# =============================================================================

PATH_PATTERN = re.compile(r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$')
COORD_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)')
COORD_EXISTS_PATTERN = re.compile(r'\(\d+,\s*\d+\)')


def extract_answer_block(text: str) -> str | None:
    if "---start_answer---" not in text or "---end_answer---" not in text:
        return None
    try:
        return text.split("---start_answer---")[1].split("---end_answer---")[0].strip()
    except IndexError:
        return None


def parse_meta(meta):
    if isinstance(meta, str):
        return json.loads(meta)
    return meta


def parse_path(answer_block: str) -> list[tuple[int, int]] | None:
    coords = COORD_PATTERN.findall(answer_block)
    if not coords:
        return None
    return [(int(x), int(y)) for x, y in coords]


def verify_format(pred, answer, meta) -> float:
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    for marker in markers:
        if pred.count(marker) != 1:
            return 0.0
    
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        return 0.0
    
    return 1.0


def verify_answer(pred, answer, meta) -> float:
    meta = parse_meta(meta)
    maze = meta["question"]
    height = meta["height"]
    width = meta["width"]
    
    answer_block = extract_answer_block(pred)
    if answer_block is None:
        return 0.0
    
    score = 0.0
    
    if "not exist" in str(answer):
        if answer_block.lower().strip() == "not exist":
            return 1.50
        elif "not exist" in answer_block.lower():
            extra = answer_block.lower().replace("not exist", "").strip()
            return 0.20 - len(extra.split()) * 0.03
        else:
            return -0.30

    if PATH_PATTERN.match(answer_block):
        score += 0.15
    else:
        letter_count = sum(1 for c in answer_block if c.isalpha())
        if letter_count > 0:
            score -= min(0.10 + letter_count * 0.02, 0.30)
        
        if not COORD_EXISTS_PATTERN.search(answer_block):
            return score - 0.25
    
    coordinate_list = parse_path(answer_block)
    if not coordinate_list:
        return score - 0.20
    
    path_length = len(coordinate_list)
    
    if path_length <= 2:
        score -= 0.40
    elif path_length <= 4:
        score -= 0.20
    elif path_length < 5:
        score -= 0.10
    
    if coordinate_list[0] != (1, 1):
        return score - 0.15
    
    score += 0.05
    
    valid_steps = 0
    error_type = None
    
    for i in range(1, len(coordinate_list)):
        curr = coordinate_list[i]
        prev = coordinate_list[i - 1]
        
        if not (1 <= curr[0] <= height and 1 <= curr[1] <= width):
            error_type = 'bounds'
            break
        
        move_distance = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
        if move_distance != 1:
            error_type = 'jump'
            break
        
        if maze[curr[0]-1][curr[1]-1] == 'B':
            error_type = 'obstacle'
            break
        
        valid_steps += 1
    
    if valid_steps > 0:
        last_valid_pos = coordinate_list[min(valid_steps, len(coordinate_list) - 1)]
        start_dist = (height - 1) + (width - 1)
        current_dist = abs(last_valid_pos[0] - height) + abs(last_valid_pos[1] - width)
        progress = 1 - (current_dist / start_dist)
        score += (progress ** 2) * 0.80
    
    if error_type == 'bounds':
        score -= 0.15
    elif error_type == 'jump':
        score -= 0.12
    elif error_type == 'obstacle':
        score -= 0.05
    
    if error_type is None:
        score += 0.10
        if coordinate_list[-1] == (height, width):
            score += 1.50
        else:
            score -= 0.10
    
    return score


def verify(pred, answer, meta) -> float:
    """Main verification function combining format and answer checks"""
    format_score = verify_format(pred, answer, meta)
    answer_score = verify_answer(pred, answer, meta)
    
    if format_score == 0.0:
        return -1.0 + answer_score * 0.3
    
    return format_score + answer_score


# =============================================================================
# VLLM VLM CLASS (from utils/vllm_utils.py)
# =============================================================================

class VLLMVLM:
    """VLLM backend for Vision-Language Models on CUDA GPUs (Qwen3-VL, etc.)
    
    Supports both text-only and image+text inference using vLLM.
    """
    
    def __init__(
        self, 
        model: str, 
        dtype: str = "auto", 
        max_model_len: Optional[int] = None, 
        gpu_memory_utilization: float = 0.9, 
        tensor_parallel_size: int = 1, 
        limit_mm_per_prompt: Optional[dict] = None,
        **kwargs
    ):
        from vllm import LLM as _VLLM, SamplingParams
        from transformers import AutoProcessor
        
        self.model_name = model
        self.SamplingParams = SamplingParams
        
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 4}
        
        self.llm = _VLLM(
            model=model, 
            dtype=dtype, 
            max_model_len=max_model_len, 
            gpu_memory_utilization=gpu_memory_utilization, 
            tensor_parallel_size=tensor_parallel_size, 
            trust_remote_code=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    
    def _load_image(self, image_path: str):
        from PIL import Image
        import requests
        from io import BytesIO
        
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_path).convert("RGB")
    
    def _build_messages(self, prompt: str, image_path: Optional[str] = None) -> List[dict]:
        content = []
        
        if image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_path if image_path.startswith(("http://", "https://")) else f"file://{image_path}"}
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def predict_batch(
        self, 
        prompts: List[str], 
        max_output_tokens: int = 512, 
        temperature: float = 0.0, 
        top_p: float = 0.95, 
        top_k: int = 20, 
        stop: Optional[List[str]] = None, 
        batch_size: int = 8,
        images: Optional[List[Optional[str]]] = None,
        **kwargs
    ) -> List[str]:
        if images is None:
            images = [None] * len(prompts)
        
        assert len(images) == len(prompts), "images list must match prompts length"
        
        sampling_params = self.SamplingParams(
            max_tokens=max_output_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            stop=stop
        )
        
        all_inputs = []
        for prompt, image in zip(prompts, images):
            messages = self._build_messages(prompt, image)
            
            formatted_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            if image is not None:
                pil_image = self._load_image(image)
                all_inputs.append({
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": pil_image}
                })
            else:
                all_inputs.append({"prompt": formatted_prompt})
        
        outputs = self.llm.generate(all_inputs, sampling_params)
        
        return [output.outputs[0].text for output in outputs]


class VLLM:
    """VLLM backend for text-only LLMs on CUDA GPUs"""
    
    def __init__(
        self, 
        model: str, 
        dtype: str = "auto", 
        max_model_len: Optional[int] = None, 
        gpu_memory_utilization: float = 0.9, 
        tensor_parallel_size: int = 1, 
        **kwargs
    ):
        from vllm import LLM as _VLLM, SamplingParams
        
        self.model_name = model
        self.SamplingParams = SamplingParams
        
        self.llm = _VLLM(
            model=model, 
            dtype=dtype, 
            max_model_len=max_model_len, 
            gpu_memory_utilization=gpu_memory_utilization, 
            tensor_parallel_size=tensor_parallel_size, 
            trust_remote_code=True,
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
    
    def predict_batch(
        self, 
        prompts: List[str], 
        max_output_tokens: int = 512, 
        temperature: float = 0.0, 
        top_p: float = 0.95, 
        top_k: int = 20, 
        stop: Optional[List[str]] = None, 
        **kwargs
    ) -> List[str]:
        sampling_params = self.SamplingParams(
            max_tokens=max_output_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            stop=stop, 
            **kwargs
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


# =============================================================================
# 25 SEED PROMPTS FROM REGENESIS PAPER (Table 8)
# =============================================================================

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
# REGENESIS PROMPTS
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
# REGENESIS PIPELINE
# =============================================================================

class ReGenesisDatasetPipeline:
    """ReGenesis-style dataset generation pipeline for maze reasoning."""
    
    def __init__(self, model_callable, verifier_callable):
        self.model = model_callable
        self.verifier = verifier_callable
        
    def _batch_generate(self, prompts: list[str]) -> list[str]:
        return self.model(prompts)
    
    def generate_dataset(
        self,
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
            puzzle_count: Number of puzzles to generate
            difficulty: Puzzle difficulty (easy, medium, hard)
            samples_per_puzzle: Number of different reasoning paths per puzzle
            use_hints_for_failures: Whether to regenerate with hints when initial attempt fails
            min_reward_threshold: Minimum reward to consider a solution correct
            batch_size: Batch size for model inference
            debug: If True, returns (results_df, debug_df) with ALL attempts including failures
            
        Returns:
            If debug=False: DataFrame with successful solutions only
            If debug=True: Tuple of (results_df, debug_df) where debug_df contains ALL attempts
        """
        # Step 1: Generate puzzles
        print(f"Step 1: Generating {puzzle_count} {difficulty} puzzles...")
        puzzles_df = generate_puzzles(count=puzzle_count, difficulty=difficulty)
        print(f"  Generated {len(puzzles_df)} puzzles")
        
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

        # Step 3: Batch adapt guidance
        print("Step 3: Adapting guidance (batched)...")
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
        
        # Step 4: Batch generate structures
        print("Step 4: Generating reasoning structures (batched)...")
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
        
        # Step 5: Batch generate reasoning paths
        print("Step 5: Generating reasoning paths (batched)...")
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
        
        # Step 6: Verify and filter
        print("Step 6: Verifying solutions...")
        results = []
        failed_items = []

        print(f"  Example Reasoning Path:\n{reasoning_paths[0][:500]}...\n")
        
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
        
        # Step 7: Retry failed items with hints
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

        # Build debug dataframe with ALL attempts
        if debug:
            debug_records = []
            for i, item in enumerate(work_items):
                first_try_success = any(
                    r['prompt'] == item['maze_prompt'] and 
                    r['seed_prompt_idx'] == item['seed_idx'] and 
                    not r.get('used_hint', False)
                    for r in results
                )
                hint_success = any(
                    r['prompt'] == item['maze_prompt'] and 
                    r['seed_prompt_idx'] == item['seed_idx'] and 
                    r.get('used_hint', False)
                    for r in results
                )
                
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
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":
    # Install dependencies first (run this in a separate cell):
    # !pip install vllm transformers pandas tqdm pillow
    
    print("=" * 60)
    print("ReGenesis SFT Dataset Generation Pipeline")
    print("=" * 60)
    
    # Initialize VLM model (use VLLM for text-only, VLLMVLM for vision)
    # For maze task, text-only is sufficient
    print("\nLoading model...")
    llm = VLLMVLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2
    )

    #     llm = VLLM(
    #     model="Qwen/Qwen2.5-7B-Instruct",  # Or use VLLMVLM for VLM
    #     max_model_len=4096,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=2  # Use both T4 GPUs!
    # )
    
    
    def model_fn(prompts: list[str]) -> list[str]:
        return llm.predict_batch(
            prompts, 
            max_output_tokens=1024, 
            temperature=0.7,
        )
    
    pipeline = ReGenesisDatasetPipeline(
        model_callable=model_fn,
        verifier_callable=verify
    )
    
    # Run pipeline
    print("\nRunning pipeline...")
    df, debug_df = pipeline.generate_dataset(
        puzzle_count=10,         # Start small for testing
        difficulty='medium',
        samples_per_puzzle=3,
        batch_size=32,
        debug=True
    )
    
    # Save to Kaggle output
    output_path = KAGGLE_OUTPUT_PATH
    df.to_parquet(f'{output_path}maze_sft_regenesis.parquet')
    debug_df.to_parquet(f'{output_path}maze_sft_regenesis_debug.parquet')
    
    print(f"\nSaved results to {output_path}")
    print(f"  - maze_sft_regenesis.parquet ({len(df)} rows)")
    print(f"  - maze_sft_regenesis_debug.parquet ({len(debug_df)} rows)")
