import re
import json


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
    Simple answer validation.
    Returns: 0 if wrong, 1 if correct
    """
    meta = parse_meta(meta)
    correct_count = meta["count"]
    
    answer_block = extract_answer_block(pred)
    if answer_block is None:
        return 0
    
    numbers = re.findall(r'\b(\d+)\b', answer_block)
    if not numbers:
        return 0
    
    final_answer = int(numbers[0])
    return 1 if final_answer == correct_count else 0


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
    """Verify format with granular scoring."""
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
    """Verify answer with relative error scoring."""
    meta = parse_meta(meta)
    correct_count = meta["count"]
    
    answer_block = extract_answer_block(pred)
    if answer_block is None:
        return -0.50
    
    # Try to extract a number
    numbers = re.findall(r'\b(\d+)\b', answer_block)
    if not numbers:
        return -0.30
    
    final_answer = int(numbers[0])
    
    # Exact match - big reward!
    if final_answer == correct_count:
        return 1.50
    
    diff = abs(final_answer - correct_count)
    
    # Relative error scoring
    # Use max(correct, 1) to avoid division by zero
    # Use max(correct, final_answer, 1) to handle overcounting fairly
    max_val = max(correct_count, final_answer, 1)
    relative_error = diff / max_val  # 0 to 1+
    
    # Convert to score: 1.0 - relative_error, clamped to [-0.3, 1.0]
    # 0% error → 1.0, 50% error → 0.5, 100% error → 0.0, >100% → negative
    score = 1.0 - relative_error
    
    # Clamp to reasonable range
    return max(-0.30, min(1.0, score))


def verify(pred, answer, meta) -> float:
    """Main verification function combining format and answer checks."""
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
