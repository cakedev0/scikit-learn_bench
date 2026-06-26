import json
import subprocess as sp
import sys
import tempfile
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

from ..config import BenchCase
from ..utils.common import hash_from_json_repr


def generate_runner_command(
    bench_case: BenchCase,
    case_file: Path,
    output_jsonl: Path,
    log_level: str,
) -> List[str]:
    command_prefix: List[str] = []
    if bench_case.bench.taskset is not None:
        command_prefix.extend(["taskset", "-c", str(bench_case.bench.taskset)])

    if bench_case.bench.distributor == "mpi":
        mpi_params = bench_case.bench.mpi_params or {}
        mpi_prefix = ["mpirun"]
        for mpi_param_name, mpi_param_value in mpi_params.items():
            mpi_prefix.extend([f"-{mpi_param_name}", str(mpi_param_value)])
        command_prefix = mpi_prefix + command_prefix

    if bench_case.bench.vtune_profiling is not None and sys.platform == "linux":
        vtune_result_dir = Path(
            bench_case.bench.vtune_results_directory or "_vtune_results"
        )
        vtune_result_dir.mkdir(parents=True, exist_ok=True)
        vtune_result_path = vtune_result_dir / "_".join(
            [
                bench_case.name(shortened=True, separator="_"),
                hash_from_json_repr(bench_case.json_dict()),
                str(int(time() * 1000)),
            ]
        )
        command_prefix = [
            "vtune",
            "-collect",
            str(bench_case.bench.vtune_profiling),
            "-r",
            str(vtune_result_path),
            "-start-paused",
            "-q",
            "-no-summary",
        ] + command_prefix

    return command_prefix + [
        sys.executable,
        "-m",
        "sklbench.runner",
        "--case-file",
        str(case_file),
        "--output-jsonl",
        str(output_jsonl),
        "--log-level",
        log_level,
    ]


def parse_runner_jsonl(output_jsonl: Path) -> List[Dict]:
    rows = []
    if not output_jsonl.exists():
        return rows
    with output_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_runner_from_case(
    bench_case: BenchCase, log_level: str
) -> Tuple[int, List[Dict], Optional[Dict]]:
    bench_case_dict = bench_case.json_dict()
    bench_time_limit = bench_case.bench.time_limit or 3600
    command_timeout = bench_time_limit * 1.5 + 10
    with tempfile.TemporaryDirectory(prefix="sklbench-run-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_file = tmp_path / "case.json"
        output_jsonl = tmp_path / "result.jsonl"
        with case_file.open("w", encoding="utf-8") as fp:
            json.dump(bench_case_dict, fp)

        command = generate_runner_command(
            bench_case, case_file, output_jsonl, log_level
        )
        try:
            result = sp.run(
                command,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                encoding="utf-8",
                timeout=command_timeout,
            )
            return_code = result.returncode
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
        except sp.TimeoutExpired as exc:
            return_code = -9
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            timeout_message = f"Command timed out after {command_timeout} seconds."
            stderr = f"{stderr}\n{timeout_message}".strip()

        rows = parse_runner_jsonl(output_jsonl)
        logs = {"stdout": stdout, "stderr": stderr}
        failed_case = None
        if return_code != 0:
            failed_case = {
                "case": bench_case_dict,
                "return_code": return_code,
                "command": command,
                "logs": logs,
            }
        for row in rows:
            row["logs"] = logs.copy()
        return return_code, rows, failed_case
