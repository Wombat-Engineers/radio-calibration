"""Run the full pipeline, then pack the results in one archive.

The pipeline runs the scenario generator (`python -m scenario`), then all notebooks of
`lab/notebooks/` in the order of their names. The configuration is the one in
`lab/configs/`.
Change the YAML files before the run to use other values, for example the values of the
"Recommended" comments on a machine with a large GPU.

Usage (from the repository root):

    uv run python lab/scripts/run_pipeline.py

The script stops at the first step that fails. It always writes
`lab/outputs/archives/<start time>.zip`, also after a failure. The archive also has the
Hydra record of the scenario run (`lab/outputs/scenario/<date>/<time>/`). The exit code is 1 if
a step failed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from utils import OUTPUTS_DIR
from utils.runs import notebook_command, pack_results, run_steps, select_notebooks

SCENARIO_RUNS = OUTPUTS_DIR / "scenario"


def _scenario_runs_since(started: datetime) -> list[Path]:
    """Return the Hydra run directories `<date>/<time>` that the scenario made after `started`."""
    if not SCENARIO_RUNS.is_dir():
        return []
    return sorted(
        path
        for path in SCENARIO_RUNS.glob("*/*")
        if path.is_dir() and path.stat().st_mtime >= started.timestamp()
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and write the archive. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    started = datetime.now()
    commands = [[sys.executable, "-m", "scenario"]]
    commands += [notebook_command(path) for path in select_notebooks([])]
    steps = run_steps(commands)
    archive = pack_results(started, steps, extra_paths=_scenario_runs_since(started))
    print(f"\nArchive: {archive}")
    return 0 if all(step.returncode == 0 for step in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
