import os
from pathlib import Path
import subprocess
import sys


def _run_with_project_environment() -> int | None:
    """Rerun this entry point with the project venv when launched elsewhere."""
    if getattr(sys, "frozen", False):
        return None

    project_root = Path(__file__).resolve().parent
    if os.name == "nt":
        project_python = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        project_python = project_root / ".venv" / "bin" / "python"

    if Path(sys.executable).resolve() == project_python.resolve():
        return None
    if not project_python.is_file():
        raise FileNotFoundError(
            f"Project Python environment not found: {project_python}. "
            "Create it and install dependencies with "
            "`python -m venv .venv` and `.venv\\Scripts\\python.exe -m pip "
            "install -r requirements.txt`."
        )

    print(
        f"Restarting Smart Waste Sorter with the project interpreter: "
        f"{project_python}",
        flush=True,
    )
    try:
        result = subprocess.run(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=project_root,
            check=False,
        )
    except KeyboardInterrupt:
        return 130
    return result.returncode


def main() -> None:
    """Launch the Smart Waste Sorter application."""
    child_status = _run_with_project_environment()
    if child_status is not None:
        raise SystemExit(child_status)

    print(f"Smart Waste Sorter Python: {sys.executable}", flush=True)

    from src.config import APP_NAME
    from src.ui import App

    app = App(title=APP_NAME)
    app.run()


if __name__ == "__main__":
    main()
