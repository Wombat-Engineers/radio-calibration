"""Run notebooks from the first cell to the last cell, then pack the results in one archive.

Use this script to run the notebooks again on data that exists. It does not run the
scenario generator: `lab/scripts/run_pipeline.py` does.

Usage (from the repository root):

    uv run python lab/scripts/run_notebooks.py
    uv run python lab/scripts/run_notebooks.py lab/notebooks/04_evaluation.ipynb

Each notebook writes a copy with its outputs to `lab/outputs/notebooks/`. The script stops
at the first notebook that fails. It always writes `lab/outputs/archives/<start time>.zip`,
also after a failure. The exit code is 1 if a notebook failed.
"""

import argparse
import sys
from datetime import datetime

from utils.runs import notebook_command, pack_results, run_steps, select_notebooks


def main(argv: list[str] | None = None) -> int:
    """Run the notebooks and write the archive. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Paths of the notebooks to run, in this order. Default: all notebooks, by name.",
    )
    args = parser.parse_args(argv)
    try:
        notebooks = select_notebooks(args.notebooks)
    except ValueError as error:
        parser.error(str(error))
    started = datetime.now()
    steps = run_steps([notebook_command(path) for path in notebooks])
    archive = pack_results(started, steps, extra_paths=[])
    print(f"\nArchive: {archive}")
    return 0 if all(step.returncode == 0 for step in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
