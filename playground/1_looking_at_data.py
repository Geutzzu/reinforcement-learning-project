import sys

sys.path.insert(0, "/home/geo/rl")

from Enigmata.verifiable_tasks.tasks.car_painting.generator import generate
from Enigmata.verifiable_tasks.tasks.car_painting.verifier import verify

puzzle = next(generate(count=1, difficulty="easy"))


from main.enigmata import generate_puzzles, verify, AVAILABLE_TASKS

df = generate_puzzles("maze", count=1000, difficulty="hard")

output_path = "/home/geo/rl/data"













