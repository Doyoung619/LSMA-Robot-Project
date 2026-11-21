"""Inspect or combine versioned rollout containers without robot SDKs."""
import argparse
import csv
import json
from pathlib import Path
from rita_container import flatten_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    runs = sorted({m.parent for path in args.paths for m in
                   ([path / "manifest.json"] if (path / "manifest.json").is_file()
                    else path.rglob("manifest.json"))})
    if not runs: parser.error("No rollout manifest.json found")
    rows = flatten_runs(runs, recompute=args.recompute)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()} for row in rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__": main()
