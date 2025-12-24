import re
import json


def verify(pred, answer, meta):
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
