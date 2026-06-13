#!/usr/bin/env python3
"""Benchmark accelerator registry selection, replay, and CPU fallback behavior.

This script measures the shared H3 contract layer and a NumPy-backed validation
adapter. It does not claim a vendor GPU/TPU/FPGA/ASIC speedup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pymars.accelerator_validation import run_benchmarks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark accelerator selection and fallback behavior."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of repeated measurements per request state.",
    )
    parser.add_argument(
        "--requested",
        default="cpu,array-test,cuda,metal,tpu",
        help="Comma-separated requested backend names or empty entries.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the validation benchmark and print summaries."""
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    requested_backends = [item.strip() or None for item in args.requested.split(",")]
    import pymars.accelerator as accelerator_module
    import pymars.runtime as runtime_module
    from pymars.accelerator_backends import ArrayModuleAcceleratorBackend

    accelerator_module.register_accelerator_backend(
        ArrayModuleAcceleratorBackend(
            name="array-test",
            marker_module="numpy",
            device_kind="array-api-validation",
        )
    )
    for row in run_benchmarks(
        runtime_module=runtime_module,
        accelerator_module=accelerator_module,
        iterations=args.iterations,
        requested_backends=requested_backends,
    ):
        print(
            "requested={requested} selected={selected} fallback={fallback} "
            "registry_median_us={registry_median_us:.2f} "
            "accelerated_predict_median_us={accelerated_predict_median_us:.2f} "
            "cpu_predict_median_us={cpu_predict_median_us:.2f}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
