#!/usr/bin/env python3
"""Run fast, fail-open syntax checks after Cursor edits supported file types."""

import ast
import json
from pathlib import Path
import sys


def validate(path: Path) -> None:
    if not path.is_file():
        return

    if path.suffix == ".py":
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    elif path.suffix == ".json" or path.suffix == ".ipynb":
        json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        file_path = payload.get("file_path")
        if file_path:
            validate(Path(file_path))
    except Exception as error:
        # afterFileEdit cannot undo an edit. Surface the error without breaking
        # the agent's workflow so it can repair the file in the next action.
        print(f"Post-edit validation warning: {error}", file=sys.stderr)

    print("{}")


if __name__ == "__main__":
    main()
