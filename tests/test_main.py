from pathlib import Path
from types import SimpleNamespace

import main


def test_launcher_restarts_with_project_venv_when_started_from_other_python(
    monkeypatch,
):
    project_root = Path(main.__file__).resolve().parent
    project_python = project_root / ".venv" / "Scripts" / "python.exe"
    invoked = {}

    def fake_run(command, *, cwd, check):
        invoked.update(command=command, cwd=cwd, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        main,
        "sys",
        SimpleNamespace(executable=r"C:\Python314\python.exe", argv=["main.py"]),
    )
    monkeypatch.setattr(main.subprocess, "run", fake_run)

    status = main._run_with_project_environment()

    assert status == 0
    assert invoked == {
        "command": [str(project_python), str(Path(main.__file__).resolve())],
        "cwd": project_root,
        "check": False,
    }


def test_launcher_does_not_restart_when_already_using_project_venv(monkeypatch):
    project_root = Path(main.__file__).resolve().parent
    project_python = project_root / ".venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(
        main,
        "sys",
        SimpleNamespace(executable=str(project_python), argv=["main.py"]),
    )

    assert main._run_with_project_environment() is None


def test_launcher_does_not_restart_when_frozen(monkeypatch):
    monkeypatch.setattr(main.sys, "frozen", True, raising=False)

    assert main._run_with_project_environment() is None


def test_launcher_exits_cleanly_when_interrupted(monkeypatch):
    monkeypatch.setattr(
        main,
        "sys",
        SimpleNamespace(executable=r"C:\Python314\python.exe", argv=["main.py"]),
    )

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.subprocess, "run", interrupt)

    assert main._run_with_project_environment() == 130
