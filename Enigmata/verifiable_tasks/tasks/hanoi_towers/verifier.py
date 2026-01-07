import re
import json

# Regex patterns
MOVE_PATTERN = re.compile(r'([ABC])\s*->\s*([ABC])')


def has_complete_format(text: str) -> bool:
    """Check if all 4 markers are present."""
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    return all(marker in text for marker in markers)


def extract_answer_block(text: str) -> str | None:
    """Extract content between answer markers."""
    if "---start_answer---" not in text or "---end_answer---" not in text:
        return None
    
    try:
        return text.split("---start_answer---")[1].split("---end_answer---")[0].strip()
    except IndexError:
        return None


def extract_reasoning_block(text: str) -> str | None:
    """Extract content between reasoning markers."""
    if "---start_reasoning---" not in text or "---end_reasoning---" not in text:
        return None
    
    try:
        return text.split("---start_reasoning---")[1].split("---end_reasoning---")[0].strip()
    except IndexError:
        return None


def parse_meta(meta):
    """Parse meta from string or dict."""
    if isinstance(meta, str):
        return json.loads(meta)
    return meta


def parse_moves(answer_block: str) -> list[tuple[str, str]] | None:
    """Parse moves from answer block."""
    if not answer_block:
        return None
    
    moves = MOVE_PATTERN.findall(answer_block)
    if not moves:
        return None
    
    return [(src, tgt) for src, tgt in moves]


def simulate_moves(initial_state: dict, moves: list[tuple[str, str]], num_disks: int):
    """
    Simulate moves and return result.
    Returns: (final_state, valid_moves_count, error_type, error_move_idx)
    """
    state = {peg: list(disks) for peg, disks in initial_state.items()}
    
    for i, (src, tgt) in enumerate(moves):
        # Check source peg has disks
        if not state[src]:
            return state, i, 'empty_source', i
        
        top_disk = state[src][-1]
        
        # Check target peg allows this disk
        if state[tgt] and state[tgt][-1] < top_disk:
            return state, i, 'larger_on_smaller', i
        
        # Valid move
        state[src].pop()
        state[tgt].append(top_disk)
    
    return state, len(moves), None, None


def is_goal_state(state: dict, num_disks: int) -> bool:
    """Check if all disks are on peg C."""
    expected = list(range(num_disks, 0, -1))
    return state['C'] == expected and not state['A'] and not state['B']


# =============================================================================
# SIMPLE VARIANTS (Binary rewards)
# =============================================================================

def verify_format_simple(pred, answer, meta):
    """
    Simple format check - validates presence of all 4 markers.
    Returns: -1 if missing, 0 if present
    """
    if not has_complete_format(pred):
        return -1
    return 0


def verify_answer_simple(pred, answer, meta):
    """
    Simple answer validation - validates solution correctness.
    Returns: 0 if wrong, 1 if correct
    """
    meta = parse_meta(meta)
    answer_block = extract_answer_block(pred)
    
    if answer_block is None:
        return 0
    
    moves = parse_moves(answer_block)
    if moves is None:
        return 0
    
    initial_state = meta["initial_state"]
    num_disks = meta["num_disks"]
    
    final_state, valid_count, error_type, _ = simulate_moves(initial_state, moves, num_disks)
    
    if error_type is not None:
        return 0
    
    if is_goal_state(final_state, num_disks):
        return 1
    
    return 0


def verify_simple(pred, answer, meta):
    """
    Simple verification combining format and answer checks.
    Returns: -1 (format fail), 0 (format ok, answer wrong), 1 (all correct)
    """
    return verify_format_simple(pred, answer, meta) + verify_answer_simple(pred, answer, meta)


# =============================================================================
# GRANULAR VERIFICATION (Shaped rewards)
# =============================================================================

def verify_format(pred, answer, meta) -> float:
    """
    Verify format with granular scoring.
    Returns: 0.0 to 1.0
    """
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    # Check each marker appears exactly once
    for marker in markers:
        if pred.count(marker) != 1:
            return 0.0
    
    # Check markers are in correct order
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        return 0.0
    
    return 1.0


def verify_answer(pred, answer, meta) -> float:
    """
    Verify answer with granular, shaped rewards.
    """
    meta = parse_meta(meta)
    answer_block = extract_answer_block(pred)
    
    if answer_block is None:
        return -0.30
    
    score = 0.0
    
    # Parse moves
    moves = parse_moves(answer_block)
    if moves is None:
        # Check if there's garbage text
        if len(answer_block) > 0:
            letter_count = sum(1 for c in answer_block if c.isalpha() and c not in 'ABC')
            if letter_count > 0:
                return -0.20 - min(letter_count * 0.02, 0.20)
        return -0.25
    
    # Good: parsed some moves
    score += 0.10
    
    initial_state = meta["initial_state"]
    num_disks = meta["num_disks"]
    optimal_length = meta.get("optimal_length", 2**num_disks - 1)
    
    # Simulate the moves
    final_state, valid_count, error_type, error_idx = simulate_moves(initial_state, moves, num_disks)
    
    # Reward for valid moves
    if valid_count > 0:
        # Progress reward: how many valid moves were made
        move_progress = min(valid_count / optimal_length, 1.0)
        score += move_progress * 0.40
    
    # Penalty for errors
    if error_type == 'empty_source':
        score -= 0.15
    elif error_type == 'larger_on_smaller':
        score -= 0.20  # This is the core rule violation
    
    # Check progress toward goal
    disks_on_c = len(final_state['C'])
    if disks_on_c > 0:
        goal_progress = disks_on_c / num_disks
        score += goal_progress * 0.30
    
    # Check if solved
    if error_type is None and is_goal_state(final_state, num_disks):
        score += 1.50  # Big bonus for solving!
        
        # Efficiency bonus for optimal or near-optimal solutions
        if len(moves) == optimal_length:
            score += 0.30  # Perfect!
        elif len(moves) <= optimal_length * 1.5:
            score += 0.15  # Good
        elif len(moves) <= optimal_length * 2:
            score += 0.05  # Acceptable
        # Penalty for very inefficient solutions
        elif len(moves) > optimal_length * 3:
            score -= 0.10
    elif error_type is None:
        # No errors but didn't reach goal
        score -= 0.10
    
    return score


def verify(pred, answer, meta) -> float:
    """
    Main verification function combining format and answer checks.
    """
    format_score = verify_format(pred, answer, meta)
    answer_score = verify_answer(pred, answer, meta)
    
    if format_score == 0.0:
        return -1.0 + answer_score * 0.3
    
    return format_score + answer_score


# =============================================================================
# REWARD FUNCTIONS REGISTRY
# =============================================================================

REWARD_FUNCTIONS = {
    # Main functions
    "verify": verify,
    "verify_simple": verify_simple,
    
    # Granular components
    "verify_format": verify_format,
    "verify_answer": verify_answer,
    "verify_format_simple": verify_format_simple,
    "verify_answer_simple": verify_answer_simple,
}
