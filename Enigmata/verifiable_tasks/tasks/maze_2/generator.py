import random
import collections
import json
from .template import PROMPT_TEMPLATE
from tqdm import tqdm

# Constants
DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]


def generate_maze(width, height, percentage=0.35):
    """Generate a random maze with obstacles."""
    maze = [['.' for _ in range(width)] for _ in range(height)]
    num_obstacles = int((width * height) * percentage)
    
    for _ in range(num_obstacles):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        maze[y][x] = 'B'
    
    maze[0][0] = 'S'
    maze[height-1][width-1] = 'E'
    return maze


def maze_to_str(maze):
    rows = [" ".join(row) for row in maze]
    return "\n".join(rows)


def compute_distances_from_goal(maze):
    """BFS from goal to compute shortest distance for each cell."""
    rows, cols = len(maze), len(maze[0])
    goal = (rows - 1, cols - 1)
    
    distances = {goal: 0}
    queue = collections.deque([goal])
    
    while queue:
        x, y = queue.popleft()
        for dx, dy, _ in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in distances:
                if maze[nx][ny] != 'B':
                    distances[(nx, ny)] = distances[(x, y)] + 1
                    queue.append((nx, ny))
    
    return distances


def find_all_shortest_paths(maze, distances):
    """Find all shortest paths from start to goal using DFS."""
    rows, cols = len(maze), len(maze[0])
    start, goal = (0, 0), (rows - 1, cols - 1)
    
    if start not in distances:
        return []  # No path exists
    
    all_paths = []
    
    def dfs(pos, path):
        if pos == goal:
            all_paths.append(path[:])
            return
        
        current_dist = distances[pos]
        for dx, dy, _ in DIRECTIONS:
            nx, ny = pos[0] + dx, pos[1] + dy
            next_pos = (nx, ny)
            # Only move to cells that are exactly 1 step closer to goal
            if next_pos in distances and distances[next_pos] == current_dist - 1:
                path.append(next_pos)
                dfs(next_pos, path)
                path.pop()
    
    dfs(start, [start])
    return all_paths


def classify_moves(maze, pos, distances):
    """Classify all possible moves from current position."""
    rows, cols = len(maze), len(maze[0])
    current_dist = distances.get(pos, float('inf'))
    
    illegal = []      # Blocked or out of bounds
    forward = []      # Moves toward goal (distance decreases)
    backward = []     # Moves away from goal (distance increases or same)
    
    for dx, dy, direction in DIRECTIONS:
        nx, ny = pos[0] + dx, pos[1] + dy
        next_pos = (nx, ny)
        coord_str = f"({nx+1},{ny+1})"
        
        # Out of bounds
        if not (0 <= nx < rows and 0 <= ny < cols):
            illegal.append(f"{direction} {coord_str} out of bounds")
            continue
        
        # Blocked
        if maze[nx][ny] == 'B':
            illegal.append(f"{direction} {coord_str} blocked")
            continue
        
        # Check distance
        next_dist = distances.get(next_pos, float('inf'))
        if next_dist == current_dist - 1:
            forward.append((direction, coord_str, next_pos))
        else:
            backward.append((direction, coord_str))
    
    return illegal, forward, backward


def generate_step_reasoning(maze, pos, next_pos, distances, step_num):
    """Generate reasoning for a single step."""
    illegal, forward, backward = classify_moves(maze, pos, distances)
    
    lines = [f"Step {step_num}: Position ({pos[0]+1},{pos[1]+1})"]
    
    # Illegal moves
    if illegal:
        lines.append(f"- Illegal: {'; '.join(illegal)}")
    else:
        lines.append("- Illegal: none")
    
    # Backward moves
    if backward:
        back_strs = [f"{d} {c}" for d, c in backward]
        lines.append(f"- Backward: {'; '.join(back_strs)}")
    
    # Forward moves (the interesting ones)
    if len(forward) == 0:
        lines.append("- Forward: none (dead end)")
    elif len(forward) == 1:
        d, c, _ = forward[0]
        lines.append(f"- Forward: only {d} {c}")
        lines.append(f"Moving {d} to {c}.")
    else:
        forward_strs = [f"{d} {c}" for d, c, _ in forward]
        lines.append(f"- Forward: {'; '.join(forward_strs)}")
        # Find which one we chose
        chosen = next((d for d, c, p in forward if p == next_pos), "")
        chosen_coord = f"({next_pos[0]+1},{next_pos[1]+1})"
        other_strs = [c for d, c, p in forward if p != next_pos]
        if other_strs:
            lines.append(f"Both {' and '.join(forward_strs)} are equally advantageous. Proceeding {chosen} to {chosen_coord}.")
        else:
            lines.append(f"Moving {chosen} to {chosen_coord}.")
    
    return "\n".join(lines)


def generate_exhaustive_reasoning(maze, path, distances):
    """Generate complete step-by-step reasoning for a path."""
    rows, cols = len(maze), len(maze[0])
    
    parts = [f"Starting at (1,1). Goal is ({rows},{cols})."]
    
    for i in range(len(path) - 1):
        step_reasoning = generate_step_reasoning(maze, path[i], path[i+1], distances, i + 1)
        parts.append(step_reasoning)
    
    # Final step
    final = path[-1]
    parts.append(f"Step {len(path)}: Reached goal at ({final[0]+1},{final[1]+1}).")
    parts.append(f"Path complete: {len(path)} positions, {len(path)-1} moves.")
    
    return "\n\n".join(parts)


def generate_no_path_reasoning(maze, distances):
    """Generate reasoning explaining why no path exists."""
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    
    parts = [f"Starting at (1,1). Goal is ({rows},{cols})."]
    
    # Count reachable cells from start (separate BFS)
    reachable = set()
    queue = collections.deque([start])
    reachable.add(start)
    
    while queue:
        x, y = queue.popleft()
        for dx, dy, _ in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in reachable:
                if maze[nx][ny] != 'B':
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
    
    parts.append(f"Explored {len(reachable)} reachable cells from start.")
    
    if goal not in reachable:
        parts.append(f"Goal ({rows},{cols}) is not reachable from start.")
        parts.append("Obstacles form a barrier blocking all paths to the goal.")
    
    parts.append("No valid path exists.")
    
    return "\n".join(parts)


def get_answer(maze, lang='en'):
    """Generate the complete answer with reasoning."""
    distances = compute_distances_from_goal(maze)
    all_paths = find_all_shortest_paths(maze, distances)
    
    if not all_paths:
        reasoning = generate_no_path_reasoning(maze, distances)
        return f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
not exist
---end_answer---"""
    
    # Randomly select one of the shortest paths
    chosen_path = random.choice(all_paths)
    reasoning = generate_exhaustive_reasoning(maze, chosen_path, distances)
    path_str = "->".join(f"({x+1},{y+1})" for x, y in chosen_path)
    
    return f"""---start_reasoning---
{reasoning}
---end_reasoning---

---start_answer---
{path_str}
---end_answer---"""


def generate(count=100, difficulty='medium', language='en', split="train"):
    """Generate maze problems."""
    height, width = 5, 5
    prompt_template = PROMPT_TEMPLATE
    exist = {}
    dif_level = {"easy": [15, 25], "medium": [26, 40], "hard": [41, 55]}
    
    for i in tqdm(range(count)):
        while True:
            p = random.randint(dif_level[difficulty][0], dif_level[difficulty][1]) / 100
            num_obs = int(p * height * width)
            maze = generate_maze(width, height, p)
            has_str = maze_to_str(maze)
            if has_str not in exist:
                exist[has_str] = 1
                break
        
        answer = get_answer(maze, lang=language)
        yield {
            "prompt": prompt_template.format(question=maze_to_str(maze)),
            "answer": answer,
            "task_name": "maze_2",
            "ability": "logic_puzzle",
            "language": language,
            "meta": json.dumps({
                "id": f"maze_2_{difficulty}_{i}",
                "question": maze,
                "width": width,
                "height": height,
                "num_obstacles": num_obs,
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
