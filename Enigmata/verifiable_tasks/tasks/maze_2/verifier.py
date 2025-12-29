import re
import json

# Precompiled regex patterns for performance
PATH_PATTERN = re.compile(r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$')
COORD_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)')
PATH_SEARCH_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)')
COORD_EXISTS_PATTERN = re.compile(r'\(\d+,\s*\d+\)')



def extract_answer_block(text: str):
    return text.split("---start_answer---")[1].split("---end_answer---")[0].strip() if "---start_answer---" in text and "---end_answer---" in text else None

def extract_reasoning_block(text: str):
    return text.split("---start_reasoning---")[1].split("---end_reasoning---")[0].strip() if "---start_reasoning---" in text and "---end_reasoning---" in text else None


# =============================================================================
# SIMPLE VARIANTS (Binary rewards: -1, 0, 1)
# =============================================================================

def verify_format_simple(pred, answer, meta):
    """
    Simple format check - just validates presence of all 4 markers.
    
    Returns:
        -1: Missing any marker
         0: All markers present
    """
    if "---start_answer---" not in pred or "---end_answer---" not in pred or "---start_reasoning---" not in pred or "---end_reasoning---" not in pred:
        return -1
    return 0


def verify_answer_simple(pred, answer, meta):
    """
    Simple answer validation - validates path correctness.
    
    Returns:
        0: Wrong answer or can't extract
        1: Correct answer
    """
    if isinstance(meta, str):
        meta = json.loads(meta)
    elif isinstance(meta, dict):
        pass
    else:
        raise ValueError('meta should be dict or str')

    normalized_pred = extract_answer_block(pred)
    normalized_reasoning = extract_reasoning_block(pred)

    if normalized_pred is None:
        return 0
    if "not exist" in answer:
        if not isinstance(normalized_pred, str):
            return 0
        if "not exist" in normalized_pred:
            return 1
        else:
            return 0
    
    maze = meta["question"]
    height = meta["height"]
    width = meta["width"]

    if isinstance(normalized_pred, str):
        coords = COORD_PATTERN.findall(normalized_pred)  # Returns [('1', '1'), ('2', '1'), ...]
        coordinates = [(int(x), int(y)) for x, y in coords]
    elif isinstance(normalized_pred, list):
        coordinates = normalized_pred
    else:
        return 0
    try:
        coordinate_list = [(int(x), int(y)) for x, y in coordinates]
        length = len(coordinate_list)
        if coordinate_list[0][0] != 1 or coordinate_list[0][1] != 1:
            return 0
        if coordinate_list[-1][0] != height or coordinate_list[-1][1] != width:
            return 0
        for i in range(1, length):
            if coordinate_list[i][0] < 1 or coordinate_list[i][0] > height:
                return 0
            if coordinate_list[i][1] < 1 or coordinate_list[i][1] > width:
                return 0
            if maze[coordinate_list[i][0]-1][coordinate_list[i][1]-1] == 'B':
                return 0
            if abs(coordinate_list[i][1] - coordinate_list[i - 1][1]) + abs(coordinate_list[i][0] - coordinate_list[i - 1][0]) != 1:
                return 0
        return 1
    except Exception as ex:
        return 0


def verify_simple(pred, answer, meta):
    """
    Simple verification combining format and answer checks.
    
    Returns:
        -1: Format failure
         0: Format OK but answer wrong
         1: Format OK and answer correct
    """
    return verify_format_simple(pred, answer, meta) + verify_answer_simple(pred, answer, meta)


# =============================================================================
# GRANULAR VARIANTS (Continuous rewards: [-1.0, 2.0])
# =============================================================================

def verify_format(pred, answer, meta):
    """
    Simple format validation - checks:
    1. Exactly one instance of all 4 markers
    2. Markers appear in correct order
    
    Returns:
        1.0: Valid format (all markers present, correct order)
        0.0: Invalid format
    """
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    # Check exactly one of each marker
    for marker in markers:
        if pred.count(marker) != 1:
            return 0.0
    
    # Check markers appear in correct order
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        return 0.0
    
    return 1.0


def verify_answer(pred, answer, meta):
    """
    Granular answer validation for maze solving.
    
    Score Range: [-0.5, ~1.7]
    
    POSITIVE Rewards:
    - +0.15: Answer block contains ONLY valid path format (no extra tokens)
    - +0.05: Path starts at (1,1)
    - +0.25: Path length bonus (continuous, scales with valid steps)
    - +0.15: Valid step ratio bonus
    - +0.20: Progress toward goal bonus
    - +0.10: All moves are valid (adjacent, in bounds, no obstacles)
    - +1.50: Path reaches destination (5,5)
    
    NEGATIVE Penalties:
    - -0.15: Extra content in answer block
    - -0.40: Very short lazy paths
    - -0.15: Out of bounds
    - -0.12: Jump (non-adjacent move)
    - -0.05: Hit obstacle
    """
    # Parse meta and answer
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            pass
    
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    maze = meta["question"]
    height = meta["height"]
    width = meta["width"]
    
    # ========== EXTRACT ANSWER BLOCK ==========
    try:
        answer_block = pred.split("---start_answer---")[1].split("---end_answer---")[0].strip()
    except IndexError:
        # Can't extract answer block - return 0 (format function handles penalty)
        return 0.0
    
    score = 0.0
    
    # ========== ANSWER BLOCK VALIDATION ==========
    
    # Check for "not exist" case
    if "not exist" in str(answer):
        if answer_block.lower().strip() == "not exist":
            # BIG REWARD - same as completing a path!
            score += 1.50  # Match completion reward
            return score
        elif "not exist" in answer_block.lower():
            score += 0.15  # Partial credit
            extra = answer_block.lower().replace("not exist", "").strip()
            if extra:
                score -= len(extra.split()) * 0.03
            return score
        else:
            # Model said a path when it should say "not exist"
            score -= 0.30  # Penalty
            return score
    
    # Validate path format (strict: only path, no extra tokens)
    if PATH_PATTERN.match(answer_block):
        score += 0.15  # REWARD: Clean path format
    else:
        # Check for letters in answer block (valid path has no letters)
        letter_count = sum(1 for c in answer_block if c.isalpha())
        if letter_count > 0:
            score -= 0.10  # Base penalty for extra text
            score -= min(letter_count * 0.02, 0.20)  # Cap per-letter penalty
        
        # Check if there's any path-like structure at all
        if not COORD_EXISTS_PATTERN.search(answer_block):
            score -= 0.25  # PENALTY: No path in answer
            return score
    
    # ========== PATH VALIDATION ==========
    
    coords = COORD_PATTERN.findall(answer_block)
    if not coords:
        score -= 0.20
        return score
    
    coordinate_list = [(int(x), int(y)) for x, y in coords]
    path_length = len(coordinate_list)
    
    # ===== PENALIZE LAZY SHORT PATHS =====
    # A valid 5x5 maze solution needs at least 9 steps (Manhattan distance)
    # Paths with only 1-3 steps are "lazy" attempts
    MIN_REASONABLE_PATH = 5  # At least try to make progress
    
    if path_length <= 2:
        # Super lazy: just (1,1) or (1,1)->(1,2)
        score -= 0.40  # HEAVY penalty for minimal effort
    elif path_length <= 4:
        # Still lazy
        score -= 0.20  # Moderate penalty
    elif path_length < MIN_REASONABLE_PATH:
        score -= 0.10  # Mild penalty
    
    # Check starts at (1,1)
    if coordinate_list[0] == (1, 1):
        score += 0.05  # Small reward for correct start (reduced from 0.10)
    else:
        score -= 0.15  # PENALTY: Wrong start
        return score  # Can't validate further
    
    # Validate path moves - count valid steps and categorize errors
    valid_steps = 0
    total_steps = len(coordinate_list) - 1
    error_type = None  # 'bounds', 'obstacle', 'jump', or None
    
    for i in range(1, len(coordinate_list)):
        curr = coordinate_list[i]
        prev = coordinate_list[i - 1]
        
        # Check bounds first (worst error - completely invalid coordinate)
        if not (1 <= curr[0] <= height and 1 <= curr[1] <= width):
            error_type = 'bounds'
            break
        
        # Check valid move (adjacent) - jumping is a logical error
        move_distance = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
        if move_distance != 1:
            error_type = 'jump'
            break
        
        # Check not hitting obstacle - this is a "softer" error (model understands movement)
        if maze[curr[0]-1][curr[1]-1] == 'B':
            error_type = 'obstacle'
            valid_steps += 1
            break
        
        # Valid step!
        valid_steps += 1
    
    # ===== REWARD PATH LENGTH (continuous) =====
    # Linear scaling from 3 to 8+ valid steps (0 to 0.25)
    score += min(max(valid_steps - 2, 0) * 0.04, 0.25)
    
    # ===== VALID STEP RATIO =====
    # Partial credit for mostly-valid paths
    if total_steps > 0:
        valid_ratio = valid_steps / total_steps
        score += valid_ratio * 0.15  # Up to 0.15 for all-valid path
    
    # ===== PROGRESS TOWARD GOAL =====
    # Reward getting closer to destination even if path breaks
    if valid_steps > 0:
        last_valid_idx = min(valid_steps, len(coordinate_list) - 1)
        last_valid_pos = coordinate_list[last_valid_idx]
        
        start_dist = (height - 1) + (width - 1)  # 8 for 5x5 maze
        current_dist = abs(last_valid_pos[0] - height) + abs(last_valid_pos[1] - width)
        progress = 1 - (current_dist / start_dist)  # 0→1 as we approach goal
        
        score += progress * 0.20  # Up to 0.20 bonus for reaching near goal
    
    # ===== PENALIZE BY ERROR TYPE =====
    if error_type == 'bounds':
        score -= 0.15
    elif error_type == 'jump':
        score -= 0.12
    elif error_type == 'obstacle':
        score -= 0.05
    
    # ===== BIG REWARD FOR COMPLETION =====
    if error_type is None:
        score += 0.10  # Valid traversal
        
        if coordinate_list[-1] == (height, width):
            # SUCCESS! Very big reward - this is the goal!
            score += 1.50  # Increased from 0.50 - make success VERY rewarding
        else:
            score -= 0.10  # Wrong endpoint
    
    return score


def verify_both_path_and_format(pred, answer, meta):
    """
    Granular reward function for maze solving with strict format requirements.
    Combines format and answer validation.
    
    Score Range: [-1.0, 2.0]
    
    Base score: 0.0 (neutral)
    
    POSITIVE Rewards (for correct behavior):
    - +0.15: Starts with ---start_reasoning---
    - +0.10: Has all 4 markers in correct order
    - +0.10: Ends with ---end_answer--- (final non-whitespace)
    - +0.15: Answer block contains ONLY valid path format (no extra tokens)
    - +0.05: Path starts at (1,1)
    - +0.25: Path length bonus (continuous, scales with valid steps)
    - +0.15: Valid step ratio bonus
    - +0.20: Progress toward goal bonus
    - +0.10: All moves are valid (adjacent, in bounds, no obstacles)
    - +1.50: Path reaches destination (5,5)
    
    NEGATIVE Penalties (for violations):
    - -0.05 per token outside of valid tag regions
    - -0.15: Missing any required marker
    - -0.15: Extra content in answer block
    - -0.20: Completely wrong format (no markers at all)
    """
    format_score = verify_format(pred, answer, meta)
    answer_score = verify_answer(pred, answer, meta)
    
    # Clamp to [-1, 2] - allow high positive for success
    return min(2.0, max(-1.0, format_score + answer_score))


def verify(pred, answer, meta):
    return verify_answer(pred, answer, meta) + verify_format(pred, answer, meta)


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
