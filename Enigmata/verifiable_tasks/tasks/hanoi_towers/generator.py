import random
import json
from .template import PROMPT_TEMPLATE
from tqdm import tqdm


def solve_hanoi(n, source='A', target='C', auxiliary='B'):
    """Generate optimal solution for Tower of Hanoi."""
    moves = []
    
    def hanoi(n, src, tgt, aux):
        if n == 1:
            moves.append((src, tgt))
        else:
            hanoi(n - 1, src, aux, tgt)
            moves.append((src, tgt))
            hanoi(n - 1, aux, tgt, src)
    
    hanoi(n, source, target, auxiliary)
    return moves


def state_to_str(state, num_disks):
    """Convert state dict to display string."""
    lines = []
    for peg in ['A', 'B', 'C']:
        disks = state[peg]
        if disks:
            disk_str = ', '.join(str(d) for d in disks)
            lines.append(f"Peg {peg}: [{disk_str}] (bottom to top)")
        else:
            lines.append(f"Peg {peg}: [] (empty)")
    
    lines.append(f"\nDisks are numbered 1 to {num_disks}. Larger number = larger disk.")
    lines.append("A larger disk cannot be placed on a smaller disk.")
    return '\n'.join(lines)


def state_to_compact(state):
    """Convert state to compact string like A:[3,2,1] B:[] C:[]"""
    parts = []
    for peg in ['A', 'B', 'C']:
        disks = state[peg]
        if disks:
            parts.append(f"{peg}:[{','.join(str(d) for d in disks)}]")
        else:
            parts.append(f"{peg}:[]")
    return ' '.join(parts)


def apply_move(state, src, tgt):
    """Apply a single move and return new state."""
    new_state = {peg: list(disks) for peg, disks in state.items()}
    if new_state[src]:
        disk = new_state[src].pop()
        new_state[tgt].append(disk)
    return new_state


def classify_move(state, src, tgt):
    """Classify a move as valid, invalid (empty source), or illegal (larger on smaller)."""
    if not state[src]:
        return 'empty_source'
    
    top_disk = state[src][-1]
    if state[tgt] and state[tgt][-1] < top_disk:
        return 'larger_on_smaller'
    
    return 'valid'


def generate_step_reasoning(state, src, tgt, step_num):
    """Generate reasoning for a single step."""
    top_disk = state[src][-1]
    new_state = apply_move(state, src, tgt)
    
    lines = [f"Step {step_num}: Move disk {top_disk} from {src} to {tgt}"]
    lines.append(f"State: {state_to_compact(new_state)}")
    
    return '\n'.join(lines), new_state


def generate_exhaustive_reasoning(initial_state, moves, num_disks):
    """Generate complete step-by-step reasoning for a solution."""
    parts = [f"Starting with {num_disks} disks on Peg A. Goal: move all to Peg C."]
    parts.append(f"Optimal solution requires {len(moves)} moves.")
    parts.append("")
    parts.append(f"Initial: {state_to_compact(initial_state)}")
    
    state = {peg: list(disks) for peg, disks in initial_state.items()}
    
    for i, (src, tgt) in enumerate(moves):
        step_reasoning, state = generate_step_reasoning(state, src, tgt, i + 1)
        parts.append(step_reasoning)
    
    parts.append("")
    parts.append(f"Done! All {num_disks} disks now on Peg C.")
    
    return '\n'.join(parts)


def generate_no_solution_reasoning():
    """Generate reasoning explaining why no solution exists (shouldn't happen for valid Hanoi)."""
    return "No valid solution exists for this configuration."


def get_answer(initial_state, num_disks, lang='en'):
    """Generate the complete answer with reasoning."""
    # Check if already solved
    if not initial_state['A'] and not initial_state['B'] and initial_state['C'] == list(range(num_disks, 0, -1)):
        return """---start_reasoning---
All disks are already on Peg C. No moves needed.
---end_reasoning---

---start_answer---

---end_answer---"""
    
    # Standard case: all disks start on A
    moves = solve_hanoi(num_disks)
    reasoning = generate_exhaustive_reasoning(initial_state, moves, num_disks)
    moves_str = ', '.join(f"{src}->{tgt}" for src, tgt in moves)
    
    return f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
{moves_str}
---end_answer---"""


def generate(count=100, difficulty='medium', language='en', split="train"):
    """Generate Tower of Hanoi problems."""
    
    # Difficulty determines number of disks
    disk_counts = {
        "easy": [2, 3],
        "medium": [3, 4],
        "hard": [4, 5]
    }
    
    for i in tqdm(range(count)):
        # Pick random disk count for this difficulty
        num_disks = random.choice(disk_counts[difficulty])
        
        # Standard start: all disks on peg A
        initial_state = {
            'A': list(range(num_disks, 0, -1)),  # [3, 2, 1] for 3 disks
            'B': [],
            'C': []
        }
        # Note: For standard Hanoi, puzzles are uniquely determined by disk count
        # No deduplication needed (and impossible for count > num disk options)
        
        question_str = state_to_str(initial_state, num_disks)
        answer = get_answer(initial_state, num_disks, lang=language)
        optimal_moves = solve_hanoi(num_disks)
        
        yield {
            "prompt": PROMPT_TEMPLATE.format(question=question_str),
            "answer": answer,
            "task_name": "hanoi_towers",
            "ability": "logic_puzzle",
            "language": language,
            "meta": json.dumps({
                "id": f"hanoi_towers_{difficulty}_{i}",
                "num_disks": num_disks,
                "initial_state": initial_state,
                "optimal_length": len(optimal_moves),
                "optimal_moves": [f"{s}->{t}" for s, t in optimal_moves],
                "split": split,
                "type": "sequential_puzzle",
                "source_url": "auto-generated",
                "dataset_name": "hanoi_towers",
                "difficulty_level": difficulty,
                "language": language,
            }),
        }
