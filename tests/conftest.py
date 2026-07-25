import sys
from pathlib import Path

# make the repo-root "pyoma_uq" package importable regardless of how pytest is
# invoked (bare `pytest` does not add the cwd to sys.path) and without needing
# an editable install
sys.path.insert(0, str(Path(__file__).parent.parent))
