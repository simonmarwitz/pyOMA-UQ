import sys
from pathlib import Path

# make the repo-root "examples" and "model" packages importable regardless
# of how pytest is invoked (bare `pytest` does not add the cwd to sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent))
