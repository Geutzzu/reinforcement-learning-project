import re
import json

# Precompiled regex patterns for performance
PATH_PATTERN = re.compile(r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$')
COORD_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)')
PATH_SEARCH_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)')
COORD_EXISTS_PATTERN = re.compile(r'\(\d+,\s*\d+\)')

# =============================================================================
# MODULAR EXTRACTION HELPERS
# =============================================================================

def has_complete_format(text: str) -> bool:
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    return all(marker in text for marker in markers)


def extract_answer_block(text: str) -> str | None:
    if "---start_answer---" not in text or "---end_answer---" not in text:
        return None
    
    try:
        return text.split("---start_answer---")[1].split("---end_answer---")[0].strip()
    except IndexError:
        return None


def extract_reasoning_block(text: str) -> str | None:
    if "---start_reasoning---" not in text or "---end_reasoning---" not in text:
        return None
    
    try:
        return text.split("---start_reasoning---")[1].split("---end_reasoning---")[0].strip()
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
# FORMAT VERIFICATION (Simple binary check) + ANSWER VERIFICATION (Granular path validation)
# =============================================================================

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
    total_steps = len(coordinate_list) - 1
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
