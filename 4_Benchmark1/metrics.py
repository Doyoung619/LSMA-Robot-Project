"""Compatibility imports for the shared benchmark/rollout metric definitions."""
from pathlib import Path
import sys

_container=Path(__file__).resolve().parents[1]/"6_container"
if str(_container) not in sys.path:sys.path.insert(0,str(_container))

from rita_container.metrics import (evaluate_trajectory as evaluate, derivatives,
                                    beat_windows, seed_distances)

__all__=["evaluate","derivatives","beat_windows","seed_distances"]
