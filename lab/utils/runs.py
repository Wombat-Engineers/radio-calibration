"""Steps of a run of the project and the archive of its results.

`lab/scripts/run_notebooks.py` and `lab/scripts/run_pipeline.py` use these functions. Run the
scripts on a machine with a large GPU, then copy the archive to the local machine.
"""

import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path

from omegaconf import OmegaConf

from utils.config import load_config
from utils.paths import CONFIG_DIR, OUTPUTS_DIR, PROJECT_ROOT, scenario_dirs

ARCHIVE_DIR = OUTPUTS_DIR / "archives"
# The packages that change the numbers of a run.
_PACKAGES = ("torch", "drjit", "mitsuba", "sionna-rt", "numpy")
_ARCHIVE_NAME = re.compile(r"[\w.-]+")


@dataclass(frozen=True)
class Step:
    """One command of a run.

    Attributes:
        command: The command and its arguments.
        seconds: The run time.
        returncode: The exit code. 0 shows success.
    """

    command: tuple[str, ...]
    seconds: float
    returncode: int


def result_paths(scenario: str) -> tuple[Path, ...]:
    """Return the results of a run of a scenario: its data, its outputs, and the configuration.

    The figure and table directories come from the configuration, not from this function.
    """
    dirs = scenario_dirs(scenario)
    return (dirs.raw, dirs.processed, dirs.outputs, CONFIG_DIR)


def select_notebooks(scenario: str, names: Sequence[str]) -> list[Path]:
    """Return the notebooks of a scenario to run.

    Args:
        scenario: Name of the scenario.
        names: Paths of notebooks, relative to the repository root or absolute. If there is
            no name, the function returns all notebooks of `lab/notebooks/<scenario>/`.

    Returns:
        The paths of the notebooks. With no name, they are in the order of their names.

    Raises:
        ValueError: If the scenario name is not valid, or if a name is not an `.ipynb` file
            in `lab/notebooks/<scenario>/`.
    """
    notebook_dir = scenario_dirs(scenario).notebooks.resolve()
    if not names:
        return sorted(notebook_dir.glob("*.ipynb"))
    paths = []
    for name in names:
        path = (PROJECT_ROOT / name).resolve()
        if path.suffix != ".ipynb" or path.parent != notebook_dir or not path.is_file():
            raise ValueError(f"{name} is not a notebook in {notebook_dir}.")
        paths.append(path)
    return paths


def notebook_command(path: Path) -> list[str]:
    """Return the command that runs all cells of a notebook, as `task notebook` does.

    The command writes a copy of the notebook with the outputs to
    `lab/outputs/<scenario>/notebooks/`. The scenario is the name of the notebook directory.
    """
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--output-dir",
        str(scenario_dirs(path.parent.name).outputs / "notebooks"),
        str(path),
    ]


def run_steps(commands: Sequence[Sequence[str]]) -> list[Step]:
    """Run the commands in order from the repository root. Stop at the first failure.

    The commands run with no shell. Their output goes to the console.

    Args:
        commands: The commands, each with its arguments.

    Returns:
        One step for each command that ran. If a command failed, it is the last step.
    """
    steps = []
    for command in commands:
        print(f"\n>>> {' '.join(command)}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(list(command), cwd=PROJECT_ROOT, check=False)
        steps.append(Step(tuple(command), time.perf_counter() - start, completed.returncode))
        if completed.returncode != 0:
            break
    return steps


def _output(command: Sequence[str]) -> str | None:
    """Return the standard output of a command, or None if the command fails."""
    try:
        completed = subprocess.run(
            list(command), cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, object]:
    """Return the facts about the machine and the code that a result depends on.

    Returns:
        The keys `python`, `platform`, `git_commit`, `git_dirty`, `packages` (the version
        of each package), and `gpus` (name, driver, and memory of each GPU from
        `nvidia-smi`). A value is None if the function cannot read it.
    """
    status = _output(["git", "status", "--porcelain"])
    gpus = _output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _output(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "packages": {package: _version(package) for package in _PACKAGES},
        "gpus": None if gpus is None else gpus.splitlines(),
    }


def write_archive(
    destination: Path,
    name: str,
    root: Path,
    paths: Sequence[Path],
    files: Mapping[str, str],
) -> Path:
    """Write a zip archive with one top folder.

    Args:
        destination: Directory of the archive. The function makes it if necessary.
        name: Name of the archive and of its top folder. Letters, digits, `_`, `.`, and
            `-` only.
        root: The archive keeps the paths relative to this directory.
        paths: Files and directories in `root`. The archive gets each file of a directory.
            The function ignores a path that does not exist.
        files: Text files to add, as {path in the top folder: text}.

    Returns:
        The path of the archive, `destination / f"{name}.zip"`.

    Raises:
        ValueError: If `name` has a character that is not permitted, or if a path is not
            in `root`.
    """
    if not _ARCHIVE_NAME.fullmatch(name):
        raise ValueError(f"The archive name {name!r} has a character that is not permitted.")
    root = root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in paths:
            path = path.resolve()
            if not path.exists():
                continue
            members = (
                [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            )
            for member in members:
                package.write(member, f"{name}/{member.relative_to(root).as_posix()}")
        for member, text in files.items():
            package.writestr(f"{name}/{member}", text)
    return archive


def pack_results(
    scenario: str, started: datetime, steps: Sequence[Step], extra_paths: Sequence[Path]
) -> Path:
    """Write the results of a run to `lab/outputs/archives/<start time>_<scenario>.zip`.

    The archive has the paths of `result_paths`, the figure and the table directories of
    the configuration, `extra_paths`, and two text files:

    - `config.yaml`: the composed configuration that the notebooks read.
    - `manifest.json`: the start and end time, the status (`ok` or `failed`), the steps,
      and `environment_info`.

    Args:
        scenario: Name of the scenario.
        started: Start time of the run.
        steps: The steps from `run_steps`.
        extra_paths: Other files or directories in the repository.

    Returns:
        The path of the archive.
    """
    cfg = load_config(f"scenario={scenario}")
    output_dirs = [PROJECT_ROOT / cfg.output.figures.dir, PROJECT_ROOT / cfg.output.tables.dir]
    manifest = {
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if all(step.returncode == 0 for step in steps) else "failed",
        "steps": [asdict(step) for step in steps],
        "environment": environment_info(),
    }
    return write_archive(
        ARCHIVE_DIR,
        f"{started.strftime('%Y%m%d-%H%M%S')}_{scenario}",
        PROJECT_ROOT,
        [*result_paths(scenario), *output_dirs, *extra_paths],
        {
            "config.yaml": OmegaConf.to_yaml(cfg, resolve=True),
            "manifest.json": json.dumps(manifest, indent=2),
        },
    )
