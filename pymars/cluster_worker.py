"""Command worker for explicit H4 multi-node replay adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import runtime
from ._model_spec import spec_from_json


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a cluster worker payload and return JSON-serializable results."""
    operation = payload.get("operation")
    spec_json = payload.get("spec_json")
    rows = payload.get("rows")
    if not isinstance(spec_json, str):
        raise TypeError("Worker payload must include spec_json.")
    if not isinstance(rows, list):
        raise TypeError("Worker payload must include row data.")

    spec = spec_from_json(spec_json)
    if operation == "predict":
        result = runtime.predict(spec, rows).tolist()
    elif operation == "design_matrix":
        result = runtime.design_matrix(spec, rows).tolist()
    else:
        raise ValueError(f"Unsupported cluster worker operation: {operation!r}.")
    return {"result": result}


def main(argv: list[str] | None = None) -> int:
    """Run a worker payload file and write result JSON to stdout."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m pymars.cluster_worker <payload.json>")
    payload = json.loads(Path(args[0]).read_text())
    sys.stdout.write(json.dumps(run_payload(payload)))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
