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
    rows, cols = len(maze), len(maze[0])
    
    intros = [
        "I need to find a path from S at (1,1) to E at ({},{}).".format(rows, cols),
        "Starting at (1,1), I'll explore the maze to reach ({},{}).".format(rows, cols),
        "Let me trace a path from the start S to the end E.",
        "I'll navigate from (1,1) to ({},{}) avoiding obstacles.".format(rows, cols),
    ]
    
    reasoning_parts = [random.choice(intros)]
    
    if path is None:
        # No path exists - describe the actual exploration with insight
        explored_set = set(explored_cells)
        end_x, end_y = rows - 1, cols - 1
        
        # Find the closest explored cell to the destination
        def distance_to_end(cell):
            return abs(cell[0] - end_x) + abs(cell[1] - end_y)
        
        closest_cells = sorted(explored_cells, key=distance_to_end)[:3]
        
        # Describe exploration from start
        if len(explored_cells) > 1:
            reasoning_parts.append("Exploring from (1,1), I can reach {} cells.".format(len(explored_cells)))
        
        # Describe the closest cell to destination and why it can't continue
        if closest_cells:
            closest = closest_cells[0]
            cx, cy = closest
            dist = distance_to_end(closest)
            
            reasoning_parts.append("Closest reachable cell to E is ({},{}) at distance {}.".format(cx+1, cy+1, dist))
            
            # What's blocking from this cell?
            blocked_neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] == 'B':
                        blocked_neighbors.append("({},{})".format(nx+1, ny+1))
            
            if blocked_neighbors:
                reasoning_parts.append("From ({},{}), blocked by obstacles at: {}.".format(
                    cx+1, cy+1, ", ".join(blocked_neighbors)))
        
        # Check what's around the destination
        dest_neighbors_blocked = []
        dest_neighbors_unreachable = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = end_x + dx, end_y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if maze[nx][ny] == 'B':
                    dest_neighbors_blocked.append("({},{})".format(nx+1, ny+1))
                elif (nx, ny) not in explored_set:
                    dest_neighbors_unreachable.append("({},{})".format(nx+1, ny+1))
        
        if dest_neighbors_blocked or dest_neighbors_unreachable:
            reasoning_parts.append("To reach E at ({},{}), need to pass through {} or {}.".format(
                rows, cols,
                dest_neighbors_blocked[0] if dest_neighbors_blocked else "(4,5)",
                "(5,4)" if len(dest_neighbors_blocked) < 2 else dest_neighbors_blocked[1]
            ))
            if dest_neighbors_blocked:
                reasoning_parts.append("{} blocked by B.".format(" and ".join(dest_neighbors_blocked)))
        
        # Clear conclusion
        conclusions = [
            "No valid path exists - E is cut off by obstacles.",
            "Cannot reach destination - all routes blocked.",
            "The obstacles form a barrier preventing access to E.",
        ]
        reasoning_parts.append(random.choice(conclusions))
        
    else:
        # Path exists - describe the journey with decision points
        path_len = len(path)
        
        # Describe each step (or key decision points for long paths)
        if path_len <= 6:
            # Short path - describe all steps
            steps_to_describe = list(range(1, path_len))
        else:
            # Longer path - describe start, key turns, and end approach
            steps_to_describe = [1, 2]  # Start
            # Add middle decision points (where multiple options existed)
            middle_steps = list(range(3, path_len - 2))
            if middle_steps:
                steps_to_describe.extend(random.sample(middle_steps, min(2, len(middle_steps))))
            steps_to_describe.extend([path_len - 2, path_len - 1])  # End approach
            steps_to_describe = sorted(set(steps_to_describe))
        
        for idx in steps_to_describe:
            if idx >= path_len:
                continue
            prev = path[idx - 1]
            curr = path[idx]
            
            # Find all options at this point
            options = []
            blocked = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = prev[0] + dx, prev[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] == 'B':
                        blocked.append("({},{})".format(nx+1, ny+1))
                    elif maze[nx][ny] != 'B':
                        options.append("({},{})".format(nx+1, ny+1))
            
            # Direction taken
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            direction = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}.get((dx, dy), "")
            
            # Generate contextual description
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
        
        # Final step to destination
        if path_len > 1:
            last = path[-1]
            reasoning_parts.append("Reached E at ({},{}).".format(last[0]+1, last[1]+1))
        
        # Summary
        conclusions = [
            "Path found with {} steps total.".format(path_len),
            "Successfully navigated in {} moves.".format(path_len - 1),
            "Complete path has {} coordinates.".format(path_len),
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





