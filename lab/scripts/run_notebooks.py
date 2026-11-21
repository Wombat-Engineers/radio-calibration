"""Run notebooks of one scenario from the first cell to the last cell, then pack the results.

Use this script to run the notebooks again on data that exists. It does not run the
scenario generator: `lab/scripts/run_pipeline.py` does.

Usage (from the repository root):

    uv run python lab/scripts/run_notebooks.py
    uv run python lab/scripts/run_notebooks.py --scenario munich_ablation
    uv run python lab/scripts/run_notebooks.py lab/notebooks/munich_map_errors/05_evaluation.ipynb

Each notebook writes a copy with its outputs to `lab/outputs/<name>/notebooks/`. The script
stops at the first notebook that fails. It always writes
`lab/outputs/archives/<start time>_<name>.zip`, also after a failure. The exit code is 1 if a
notebook failed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from utils import load_config
from utils.runs import notebook_command, pack_results, run_steps, select_notebooks


def main(argv: list[str] | None = None) -> int:
    """Run the notebooks and write the archive. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Paths of notebooks of one scenario, in this order. Default: all, by name.",
    )
    parser.add_argument(
        "--scenario",
        help="Name of the scenario. Default: the directory of the first notebook, else the "
        "scenario of config.yaml.",
    )
    args = parser.parse_args(argv)
    if args.scenario:
        scenario = args.scenario
    elif args.notebooks:
        scenario = Path(args.notebooks[0]).parent.name
    else:
        scenario = load_config().scenario.name
    try:
        notebooks = select_notebooks(scenario, args.notebooks)
    except ValueError as error:
        parser.error(str(error))
    started = datetime.now()
    steps = run_steps([notebook_command(path) for path in notebooks])
    archive = pack_results(scenario, started, steps)
    print(f"\nArchive: {archive}")
    return 0 if all(step.returncode == 0 for step in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
