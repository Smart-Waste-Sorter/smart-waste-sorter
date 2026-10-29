"""Run a standalone pyttsx3 check without the camera or model."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> int:
    print(f"Python: {sys.executable}", flush=True)

    try:
        import pyttsx3
    except ImportError as exc:
        print(
            f"Speech test failed: pyttsx3 is not installed ({exc}). "
            f"Install it with: \"{sys.executable}\" -m pip install pyttsx3",
            file=sys.stderr,
            flush=True,
        )
        return 1

    engine = None
    try:
        print(f"pyttsx3 version: {importlib.metadata.version('pyttsx3')}", flush=True)
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        print(f"Speech engine initialized; voices found: {len(voices)}", flush=True)
        engine.say("Smart Waste Sorter voice test. Speech is working.")
        engine.runAndWait()
    except Exception as exc:
        print(
            f"Speech test failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception as exc:
                print(
                    f"Speech engine cleanup failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        "The speech engine completed the test request. Please confirm that "
        "you heard it from the computer speakers.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
