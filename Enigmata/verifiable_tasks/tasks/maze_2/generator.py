import random
import collections
import json
from .template import PROMPT_TEMPLATE
from tqdm import tqdm

def generate_maze(width, height, percentage=0.35):
    maze = [['.' for _ in range(width)] for _ in range(height)]
    
    num_obstacles = int((width * height) * percentage)
    for _ in range(num_obstacles):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        maze[y][x] = 'B'
    
    maze[0][0] = 'S'
    maze[height-1][width-1] = 'E'
    
    return maze

def print_maze(maze):
    for row in maze:
        print(' '.join(row))

def maze_to_str(maze):
    cols = len(maze[0])
    header = "    " + " ".join(str(i+1) for i in range(cols))
    rows = [f"{i+1}   " + " ".join(row) for i, row in enumerate(maze)]
    return header + "\n" + "\n".join(rows)

def generate_reasoning(maze, path, explored_cells):
    """Generate diverse reasoning text based on actual BFS exploration."""
    rows, cols = len(maze), len(maze[0])
    
    # Random intro phrases
    intros = [
        "I need to find a path from S at (1,1) to E at ({},{}).".format(rows, cols),
        "Starting at (1,1), I'll explore the maze to reach ({},{}).".format(rows, cols),
        "Let me trace a path from the start S to the end E.",
        "I'll navigate from (1,1) to ({},{}) avoiding obstacles.".format(rows, cols),
    ]
    
    # Random exploration phrases
    explore_phrases = [
        "From ({},{}), I can move to ",
        "At ({},{}), the possible moves are ",
        "Standing at ({},{}), I see I can go to ",
        "Position ({},{}) connects to ",
    ]
    
    # Random move phrases
    move_phrases = [
        "Moving to ({},{}).",
        "I'll go to ({},{}).",
        "Proceeding to ({},{}).",
        "Taking the path to ({},{}).",
    ]
    
    # Random obstacle phrases
    obstacle_phrases = [
        "({},{}) is blocked by B.",
        "Can't go to ({},{}) - obstacle.",
        "({},{}) has an obstacle.",
    ]
    
    reasoning_parts = [random.choice(intros)]
    
    if path is None:
        # No path exists
        reasoning_parts.append("After exploring all possibilities, there's no valid path.")
        reasoning_parts.append("The obstacles block all routes to the destination.")
    else:
        # Describe some key steps (not all, to keep it reasonable)
        directions = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
        
        # Pick 2-4 key points to describe
        num_points = min(random.randint(2, 4), len(path) - 1)
        key_indices = sorted(random.sample(range(1, len(path)), num_points)) if len(path) > 1 else []
        
        for idx in key_indices:
            prev = path[idx - 1]
            curr = path[idx]
            
            # Describe the exploration at this point
            phrase = random.choice(explore_phrases).format(prev[0]+1, prev[1]+1)
            
            # Find adjacent passable cells
            adjacent = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = prev[0] + dx, prev[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] != 'B':
                        adjacent.append("({},{})".format(nx+1, ny+1))
            
            if adjacent:
                phrase += ", ".join(adjacent) + ". "
                phrase += random.choice(move_phrases).format(curr[0]+1, curr[1]+1)
                reasoning_parts.append(phrase)
        
        # Final summary
        path_len = len(path)
        conclusions = [
            "This gives a valid path of {} steps.".format(path_len),
            "Found a route with {} moves.".format(path_len - 1),
            "Successfully reached the destination in {} steps.".format(path_len),
            "The path has {} coordinates total.".format(path_len),
        ]
        reasoning_parts.append(random.choice(conclusions))
    
    return "\n".join(reasoning_parts)


def get_answer(maze, lang='en'):
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

def generate(count=100, difficulty='medium', language='en', split="train"):
    height=5
    width=5
    prompt_template = PROMPT_TEMPLATE
    exist = {}
    dif_level = {"easy" : [15,25], "medium" : [26,40], "hard" : [41,55]}
    for i in tqdm(range(count)):
        while True:
            p = random.randint(dif_level[difficulty][0], dif_level[difficulty][1])/100
            num_obs = int(p*height*width)
            maze = generate_maze(width, height, p)
            has_str = maze_to_str(maze)
            if has_str in exist:
                continue
            else:
                exist[has_str] = 1
                break
        answer = get_answer(maze, lang=language)
        yield {
            "prompt": prompt_template.format(question=maze_to_str(maze)),
            "answer":  answer,
            "task_name": "maze_2",    
            "ability": "logic_puzzle", 
            "language": language,
            "meta": json.dumps({
                "id":"maze_2_"+difficulty+'_'+str(i),
                "question": maze,
                "width": width,
                "height": height,
                "num_obstacles":num_obs,
                "answer": answer,
                "rationale": "", 
                "split": split,
                "type": "sequential_puzzle", 
                "source_url": "auto-generated", 
                "dataset_name": "maze_2", 
                "difficulty_level": difficulty,
                "language": language,
            }),
        }
