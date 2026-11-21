"""Single checkout entry point; no PYTHONPATH or numeric Python imports needed."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
for folder in (ROOT,ROOT.parent/"1_finite_version",ROOT.parent/"2_infinite_version",ROOT.parent/"6_container"):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))

if __name__ == "__main__":
    from rita_core.cli import main
    main()
