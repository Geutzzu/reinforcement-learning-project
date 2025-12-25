import re
import json

# Precompiled regex patterns for performance
PATH_PATTERN = re.compile(r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$')
COORD_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)')
PATH_SEARCH_PATTERN = re.compile(r'\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)')
COORD_EXISTS_PATTERN = re.compile(r'\(\d+,\s*\d+\)')


def verify_old(pred, answer, meta):
    score = 0.0
    
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
    
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    has_all_markers = all(m in pred for m in markers)
    
    if has_all_markers:
        score += 0.1
    else:
        return score
    
    answer_block = pred.split("---start_answer---")[1].split("---end_answer---")[0].strip()
    
    if "not exist" in str(answer):
        if "not exist" in answer_block.lower():
            return 1.0
        return score
    
    path_pattern = r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$'
    if not re.match(path_pattern, answer_block):
        return score
    
    coords = re.findall(r'\((\d+),\s*(\d+)\)', answer_block)
    if not coords:
        return score
    
    coordinate_list = [(int(x), int(y)) for x, y in coords]
    
    if coordinate_list[0] != (1, 1):
        return score + 0.1
    
    valid_path = True
    for i in range(1, len(coordinate_list)):
        curr = coordinate_list[i]
        prev = coordinate_list[i - 1]
        
        if not (1 <= curr[0] <= height and 1 <= curr[1] <= width):
            valid_path = False
            break
        

        if maze[curr[0]-1][curr[1]-1] == 'B':
            valid_path = False
            break
        
        if abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) != 1:
            valid_path = False
            break
    
    if not valid_path:
        return score + 0.2 
    
    if coordinate_list[-1] != (height, width):
        return score + 0.3 
    
    return 1.0




def verify_old_2(pred, answer, meta):
    """
    Granular reward function for maze solving with strict format requirements.
    
    Score Range: [-1.0, 1.0]
    
    Base score: 0.0 (neutral)
    
    POSITIVE Rewards (for correct behavior):
    - +0.15: Starts with ---start_reasoning---
    - +0.10: Has all 4 markers in correct order
    - +0.10: Ends with ---end_answer--- (final non-whitespace)
    - +0.15: Answer block contains ONLY valid path format (no extra tokens)
    - +0.10: Path starts at (1,1)
    - +0.10: All moves are valid (adjacent, in bounds, no obstacles)
    - +0.30: Path reaches destination (5,5)
    
    NEGATIVE Penalties (for violations):
    - -0.05 per token outside of valid tag regions
    - -0.10: Missing any required marker
    - -0.15: Extra content in answer block
    - -0.20: Completely wrong format (no markers at all)
    """
    score = 0.0
    
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
    
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    # ========== STRICT MARKER VALIDATION ==========
    
    # Count occurrences of each marker
    marker_counts = {m: pred.count(m) for m in markers}
    
    # Check for DUPLICATE markers (should be exactly 1 of each)
    has_duplicates = any(count > 1 for count in marker_counts.values())
    if has_duplicates:
        # Strong penalty for each duplicate
        duplicate_count = sum(max(0, count - 1) for count in marker_counts.values())
        score -= 0.20 * duplicate_count  # -0.20 per duplicate marker
    
    # Check for MISSING markers
    missing_markers = [m for m in markers if marker_counts[m] == 0]
    if missing_markers:
        score -= 0.15 * len(missing_markers)  # -0.15 per missing marker
        
        # If no markers at all, catastrophic
        if len(missing_markers) == 4:
            return -1.0
        
        # Can't validate further without all markers
        return max(-1.0, score)
    
    # Check markers appear in CORRECT ORDER
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        score -= 0.30  # Strong penalty for wrong order
        # Still continue to check other aspects
    else:
        score += 0.10  # Reward for correct order
    
    # Extra check: no markers should appear after ---end_answer---
    end_answer_pos = pred.find("---end_answer---")
    for m in markers[:-1]:  # All markers except end_answer
        if pred.find(m, end_answer_pos + 1) != -1:
            score -= 0.15  # Penalty for markers after end
    
    # ========== FORMAT REWARDS ==========
    
    stripped_pred = pred.strip()
    
    # Check if starts with ---start_reasoning--- (first non-whitespace)
    if stripped_pred.startswith("---start_reasoning---"):
        score += 0.15  # REWARD: Good start
    else:
        # PENALTY: Content before first marker
        before_first_marker = pred.split("---start_reasoning---")[0]
        extra_tokens = len(before_first_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # Check if ends with ---end_answer--- (last non-whitespace)
    if stripped_pred.endswith("---end_answer---"):
        score += 0.10  # REWARD: Clean ending
    else:
        # PENALTY: Content after last marker
        after_last_marker = pred.split("---end_answer---")[-1]
        extra_tokens = len(after_last_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # ========== EXTRACT SECTIONS ==========
    
    try:
        between_sections = pred.split("---end_reasoning---")[1].split("---start_answer---")[0]
        answer_block = pred.split("---start_answer---")[1].split("---end_answer---")[0].strip()
        
        # PENALTY: Excessive content between sections
        between_tokens = len(between_sections.strip().split()) if between_sections.strip() else 0
        if between_tokens > 2:  # Allow small whitespace
            score -= (between_tokens - 2) * 0.02
            
    except IndexError:
        return max(-1.0, score - 0.20)
    
    # ========== ANSWER BLOCK VALIDATION ==========
    
    # Check for "not exist" case
    if "not exist" in str(answer):
        if answer_block.lower().strip() == "not exist":
            score += 0.30  # REWARD: Correct "not exist"
            return min(1.0, max(-1.0, score))
        elif "not exist" in answer_block.lower():
            score += 0.10  # Partial credit
            # But penalize extra content
            extra = answer_block.lower().replace("not exist", "").strip()
            if extra:
                score -= len(extra.split()) * 0.03
            return min(1.0, max(-1.0, score))
        else:
            score -= 0.20  # PENALTY: Should have said "not exist"
            return max(-1.0, score)
    
    # Validate path format (strict: only path, no extra tokens)
    path_pattern = r'^\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)$'
    
    if re.match(path_pattern, answer_block):
        score += 0.15  # REWARD: Clean path format
    else:
        # Try to find path with extra content
        path_match = re.search(r'\((\d+),\s*(\d+)\)((?:->\((\d+),\s*(\d+)\))*)', answer_block)
        if path_match:
            path_text = path_match.group(0)
            extra_content = answer_block.replace(path_text, "").strip()
            if extra_content:
                extra_tokens = len(extra_content.split())
                score -= 0.15  # Base penalty for dirty answer block
                score -= extra_tokens * 0.03  # Per-token penalty
        else:
            # No valid path found at all
            score -= 0.25  # PENALTY: No path in answer
            return max(-1.0, score)
    
    # ========== PATH VALIDATION ==========
    
    coords = re.findall(r'\((\d+),\s*(\d+)\)', answer_block)
    if not coords:
        score -= 0.20
        return max(-1.0, score)
    
    coordinate_list = [(int(x), int(y)) for x, y in coords]
    
    # Check starts at (1,1)
    if coordinate_list[0] == (1, 1):
        score += 0.10  # REWARD: Correct start
    else:
        score -= 0.10  # PENALTY: Wrong start
    
    # Validate path moves
    valid_path = True
    
    for i in range(1, len(coordinate_list)):
        curr = coordinate_list[i]
        prev = coordinate_list[i - 1]
        
        # Check bounds
        if not (1 <= curr[0] <= height and 1 <= curr[1] <= width):
            valid_path = False
            score -= 0.05  # PENALTY: Out of bounds
            break
        
        # Check not hitting obstacle
        if maze[curr[0]-1][curr[1]-1] == 'B':
            valid_path = False
            score -= 0.10  # PENALTY: Hit obstacle
            break
        
        # Check valid move (adjacent)
        if abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) != 1:
            valid_path = False
            score -= 0.05  # PENALTY: Invalid move
            break
    
    if valid_path:
        score += 0.10  # REWARD: Valid path traversal
    
    # Check reaches destination
    if valid_path and coordinate_list[-1] == (height, width):
        score += 0.30  # REWARD: SUCCESS!
    elif valid_path:
        score -= 0.05  # PENALTY: Valid path but wrong endpoint
    
    # Clamp to [-3, 1] - allow deeper penalties, max at 1.0
    return min(1.0, max(-3.0, score))

    

def verify_old_3(pred, answer, meta):
    """
    Granular reward function for maze solving with strict format requirements.
    
    Score Range: [-1.0, 1.0]
    
    Base score: 0.0 (neutral)
    
    POSITIVE Rewards (for correct behavior):
    - +0.15: Starts with ---start_reasoning---
    - +0.10: Has all 4 markers in correct order
    - +0.10: Ends with ---end_answer--- (final non-whitespace)
    - +0.15: Answer block contains ONLY valid path format (no extra tokens)
    - +0.10: Path starts at (1,1)
    - +0.10: All moves are valid (adjacent, in bounds, no obstacles)
    - +0.30: Path reaches destination (5,5)
    
    NEGATIVE Penalties (for violations):
    - -0.05 per token outside of valid tag regions
    - -0.10: Missing any required marker
    - -0.15: Extra content in answer block
    - -0.20: Completely wrong format (no markers at all)
    """
    score = 0.0
    
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
    
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    # ========== STRICT MARKER VALIDATION ==========
    
    # Count occurrences of each marker
    marker_counts = {m: pred.count(m) for m in markers}
    
    # Check for DUPLICATE markers (should be exactly 1 of each)
    has_duplicates = any(count > 1 for count in marker_counts.values())
    if has_duplicates:
        # Strong penalty for each duplicate
        duplicate_count = sum(max(0, count - 1) for count in marker_counts.values())
        score -= 0.20 * duplicate_count  # -0.20 per duplicate marker
    
    # Check for MISSING markers
    missing_markers = [m for m in markers if marker_counts[m] == 0]
    if missing_markers:
        score -= 0.15 * len(missing_markers)  # -0.15 per missing marker
        
        # If no markers at all, catastrophic
        if len(missing_markers) == 4:
            return -1.0
        
        # Can't validate further without all markers
        return max(-1.0, score)
    
    # Check markers appear in CORRECT ORDER
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        score -= 0.30  # Strong penalty for wrong order
        # Still continue to check other aspects
    else:
        score += 0.10  # Reward for correct order
    
    # Extra check: no markers should appear after ---end_answer---
    end_answer_pos = pred.find("---end_answer---")
    for m in markers[:-1]:  # All markers except end_answer
        if pred.find(m, end_answer_pos + 1) != -1:
            score -= 0.15  # Penalty for markers after end
    
    # ========== FORMAT REWARDS ==========
    
    stripped_pred = pred.strip()
    
    # Check if starts with ---start_reasoning--- (first non-whitespace)
    if stripped_pred.startswith("---start_reasoning---"):
        score += 0.15  # REWARD: Good start
    else:
        # PENALTY: Content before first marker
        before_first_marker = pred.split("---start_reasoning---")[0]
        extra_tokens = len(before_first_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # Check if ends with ---end_answer--- (last non-whitespace)
    if stripped_pred.endswith("---end_answer---"):
        score += 0.10  # REWARD: Clean ending
    else:
        # PENALTY: Content after last marker
        after_last_marker = pred.split("---end_answer---")[-1]
        extra_tokens = len(after_last_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # ========== EXTRACT SECTIONS ==========
    
    try:
        between_sections = pred.split("---end_reasoning---")[1].split("---start_answer---")[0]
        answer_block = pred.split("---start_answer---")[1].split("---end_answer---")[0].strip()
        
        # PENALTY: Excessive content between sections
        between_tokens = len(between_sections.strip().split()) if between_sections.strip() else 0
        if between_tokens > 2:  # Allow small whitespace
            score -= (between_tokens - 2) * 0.02
            
    except IndexError:
        return max(-1.0, score - 0.20)
    
    # ========== ANSWER BLOCK VALIDATION ==========
    
    # Check for "not exist" case
    if "not exist" in str(answer):
        if answer_block.lower().strip() == "not exist":
            # BIG REWARD - same as completing a path!
            score += 1.50  # Match completion reward
            return min(2.0, max(-1.0, score))
        elif "not exist" in answer_block.lower():
            score += 0.15  # Partial credit
            extra = answer_block.lower().replace("not exist", "").strip()
            if extra:
                score -= len(extra.split()) * 0.03
            return min(2.0, max(-1.0, score))
        else:
            # Model said a path when it should say "not exist"
            score -= 0.30  # Penalty
            return max(-1.0, score)
    
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
            return max(-1.0, score)
    
    # ========== PATH VALIDATION ==========
    
    coords = COORD_PATTERN.findall(answer_block)
    if not coords:
        score -= 0.20
        return max(-1.0, score)
    
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
        return max(-1.0, score)  # Can't validate further
    
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
    
    # ===== REWARD PATH LENGTH (not just ratio) =====
    # Encourage LONGER paths, not just valid ones
    if valid_steps >= 8:
        score += 0.25  # Long valid path - great!
    elif valid_steps >= 5:
        score += 0.15  # Medium path - good
    elif valid_steps >= 3:
        score += 0.05  # Short but trying
    # No bonus for 1-2 step paths
    
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
    
    # Clamp to [-1, 2] - allow high positive for success
    return min(2.0, max(-1.0, score))





def verify(pred, answer, meta):
    """
    Granular reward function for maze solving with strict format requirements.
    
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
    score = 0.0
    
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
    
    markers = ["---start_reasoning---", "---end_reasoning---", "---start_answer---", "---end_answer---"]
    
    # ========== STRICT MARKER VALIDATION ==========
    
    # Count occurrences of each marker
    marker_counts = {m: pred.count(m) for m in markers}
    
    # Check for DUPLICATE markers (should be exactly 1 of each)
    has_duplicates = any(count > 1 for count in marker_counts.values())
    if has_duplicates:
        # Strong penalty for each duplicate
        duplicate_count = sum(max(0, count - 1) for count in marker_counts.values())
        score -= 0.20 * duplicate_count  # -0.20 per duplicate marker
    
    # Check for MISSING markers
    missing_markers = [m for m in markers if marker_counts[m] == 0]
    if missing_markers:
        score -= 0.15 * len(missing_markers)  # -0.15 per missing marker
        
        # If no markers at all, catastrophic
        if len(missing_markers) == 4:
            return -1.0
        
        # Can't validate further without all markers
        return max(-1.0, score)
    
    # Check markers appear in CORRECT ORDER
    positions = [pred.find(m) for m in markers]
    if positions != sorted(positions):
        score -= 0.30  # Strong penalty for wrong order
        # Still continue to check other aspects
    else:
        score += 0.10  # Reward for correct order
    
    # Extra check: no markers should appear after ---end_answer---
    end_answer_pos = pred.find("---end_answer---")
    for m in markers[:-1]:  # All markers except end_answer
        if pred.find(m, end_answer_pos + 1) != -1:
            score -= 0.15  # Penalty for markers after end
    
    # ========== FORMAT REWARDS ==========
    
    stripped_pred = pred.strip()
    
    # Check if starts with ---start_reasoning--- (first non-whitespace)
    if stripped_pred.startswith("---start_reasoning---"):
        score += 0.15  # REWARD: Good start
    else:
        # PENALTY: Content before first marker
        before_first_marker = pred.split("---start_reasoning---")[0]
        extra_tokens = len(before_first_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # Check if ends with ---end_answer--- (last non-whitespace)
    if stripped_pred.endswith("---end_answer---"):
        score += 0.10  # REWARD: Clean ending
    else:
        # PENALTY: Content after last marker
        after_last_marker = pred.split("---end_answer---")[-1]
        extra_tokens = len(after_last_marker.split())
        score -= extra_tokens * 0.05  # -0.05 per extra token
    
    # ========== EXTRACT SECTIONS ==========
    
    try:
        between_sections = pred.split("---end_reasoning---")[1].split("---start_answer---")[0]
        answer_block = pred.split("---start_answer---")[1].split("---end_answer---")[0].strip()
        
        # PENALTY: Excessive content between sections (slightly lenient)
        between_tokens = len(between_sections.strip().split()) if between_sections.strip() else 0
        if between_tokens > 5:
            score -= (between_tokens - 5) * 0.01
            
    except IndexError:
        return max(-1.0, score - 0.20)
    
    # ========== ANSWER BLOCK VALIDATION ==========
    
    # Check for "not exist" case
    if "not exist" in str(answer):
        if answer_block.lower().strip() == "not exist":
            # BIG REWARD - same as completing a path!
            score += 1.50  # Match completion reward
            return min(2.0, max(-1.0, score))
        elif "not exist" in answer_block.lower():
            score += 0.15  # Partial credit
            extra = answer_block.lower().replace("not exist", "").strip()
            if extra:
                score -= len(extra.split()) * 0.03
            return min(2.0, max(-1.0, score))
        else:
            # Model said a path when it should say "not exist"
            score -= 0.30  # Penalty
            return max(-1.0, score)
    
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
            return max(-1.0, score)
    
    # ========== PATH VALIDATION ==========
    
    coords = COORD_PATTERN.findall(answer_block)
    if not coords:
        score -= 0.20
        return max(-1.0, score)
    
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
        return max(-1.0, score)  # Can't validate further
    
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
    
    # Clamp to [-1, 2] - allow high positive for success
    return min(2.0, max(-1.0, score))