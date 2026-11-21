"""Run the full pipeline of one scenario, then pack the results in one archive.

The pipeline runs the scenario generator (`python -m scenario scenario=<name>`), then all
notebooks of `lab/notebooks/<name>/` in the order of their names. The configuration is the
one in `lab/configs/`. Change the YAML files before the run to use other values, for example
the values of the "Recommended" comments on a machine with a large GPU.

Usage (from the repository root):

    uv run python lab/scripts/run_pipeline.py
    uv run python lab/scripts/run_pipeline.py --scenario munich_new_site

The script stops at the first step that fails. It always writes
`lab/outputs/archives/<start time>_<name>.zip`, also after a failure. The archive also has the
Hydra record of the scenario run (`lab/outputs/<name>/scenario/<date>/<time>/`). The exit code
is 1 if a step failed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from utils import load_config, scenario_dirs
from utils.runs import notebook_command, pack_results, run_steps, select_notebooks


def _scenario_runs_since(scenario: str, started: datetime) -> list[Path]:
    """Return the Hydra run directories `<date>/<time>` that the scenario made after `started`."""
    runs = scenario_dirs(scenario).outputs / "scenario"
    if not runs.is_dir():
        return []
    return sorted(
        path
        for path in runs.glob("*/*")
        if path.is_dir() and path.stat().st_mtime >= started.timestamp()
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and write the archive. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", help="Name of the scenario. Default: the scenario of config.yaml."
    )
    args = parser.parse_args(argv)
    scenario = args.scenario or load_config().scenario.name
    try:
        notebooks = select_notebooks(scenario, [])
    except ValueError as error:
        parser.error(str(error))
    started = datetime.now()
    commands = [[sys.executable, "-m", "scenario", f"scenario={scenario}"]]
    commands += [notebook_command(path) for path in notebooks]
    steps = run_steps(commands)
    archive = pack_results(
        scenario, started, steps, extra_paths=_scenario_runs_since(scenario, started)
    )
    print(f"\nArchive: {archive}")
    return 0 if all(step.returncode == 0 for step in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
