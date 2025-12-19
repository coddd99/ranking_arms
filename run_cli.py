from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent  
    parent = here.parent                   
    sys.path.insert(0, str(parent))

    runpy.run_module("bandit_ranker_m1.cli", run_name="__main__")
