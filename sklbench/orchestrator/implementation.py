import json
import re
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from psutil import cpu_count
from tqdm import tqdm

from ..config import EstimatorCase
from ..datasets import load_data_with_cleanup
from ..utils.common import custom_format, hash_from_json_repr
from ..utils.logger import logger
from .commands import run_runner_from_case
from .env import get_environment_info


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def get_hardware_hash(hardware_info: dict) -> str:
    return hash_from_json_repr(hardware_info, hash_limit=6)


def get_software_hash(software_info: dict) -> str:
    return hash_from_json_repr(software_info, hash_limit=6)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _case_basename(bench_case: EstimatorCase) -> str:
    case_slug = bench_case.name(shortened=True, separator="_")
    case_slug = _UNSAFE_FILENAME_CHARS.sub("_", case_slug).strip("_")
    case_hash = hash_from_json_repr(bench_case.json_dict())
    return f"{case_slug}_{case_hash}_{_timestamp()}"


def save_environment_sidecars(
    hardware_hash: str,
    software_hash: str,
    env_info: dict,
    results_dir: str,
):
    results_root = Path(results_dir)
    hardware_env_dir = results_root / "hardware-envs"
    software_env_dir = results_root / "software-envs"
    hardware_env_dir.mkdir(parents=True, exist_ok=True)
    software_env_dir.mkdir(parents=True, exist_ok=True)

    env_files = [
        (hardware_env_dir / f"{hardware_hash}.json", env_info["hardware"]),
        (software_env_dir / f"{software_hash}.json", env_info["software"]),
    ]
    for env_file, env_content in env_files:
        try:
            with env_file.open("x", encoding="utf-8") as fp:
                json.dump(env_content, fp, indent=4)
        except FileExistsError:
            pass


def save_benchmark_record(
    record_path: Path,
    bench_case: EstimatorCase,
    rows: list[dict],
    failed_case: dict | None,
    hardware_hash: str,
    software_hash: str,
):
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "hardware_hash": hardware_hash,
        "software_hash": software_hash,
        "case": bench_case.json_dict(),
        "results": rows,
        "failed_case": failed_case,
    }
    with record_path.open("x", encoding="utf-8") as fp:
        json.dump(record, fp, indent=4)
    logger.warning(f"Benchmark result saved to {record_path}")


def call_benchmarks(
    bench_cases: list[EstimatorCase],
    hardware_hash: str,
    software_hash: str,
    results_dir: str,
    log_level: str = "WARNING",
    early_exit: bool = False,
) -> tuple[int, list[dict], list[dict]]:
    results = []
    failed_cases = []
    return_code = 0
    results_root = Path(results_dir)
    records_dir = results_root / "records"
    profiles_dir = results_root / "profiles"

    bench_cases_with_pbar = tqdm(bench_cases)
    for bench_case in bench_cases_with_pbar:
        basename = _case_basename(bench_case)
        record_path = records_dir / f"{basename}.json"
        profile_path = profiles_dir / f"{basename}.svg"
        record_saved = False
        bench_cases_with_pbar.set_description(
            custom_format(bench_case.name(shortened=True), bcolor="HEADER")
        )
        try:
            bench_return_code, rows, failed_case = run_runner_from_case(
                bench_case, log_level
            )
            save_benchmark_record(
                record_path,
                bench_case,
                rows,
                failed_case,
                hardware_hash,
                software_hash,
            )
            record_saved = True
            if bench_return_code != 0:
                return_code = bench_return_code
                if failed_case is not None:
                    failed_cases.append(failed_case)
                if early_exit:
                    break
            results.append(
                {
                    "hardware_hash": hardware_hash,
                    "software_hash": software_hash,
                    "case": bench_case.json_dict(),
                    "results": rows,
                    "failed_case": failed_case,
                }
            )
            if bench_return_code == 0 and bench_case.bench.py_spy_profiling:
                profiles_dir.mkdir(parents=True, exist_ok=True)
                profile_n_runs = max(1, bench_case.bench.n_runs // 3)
                profile_return_code, _, profile_failed_case = run_runner_from_case(
                    bench_case,
                    log_level,
                    py_spy_output=profile_path,
                    n_runs_override=profile_n_runs,
                )
                if profile_return_code != 0:
                    return_code = profile_return_code
                    if profile_failed_case is not None:
                        failed_cases.append(profile_failed_case)
                    if early_exit:
                        break
        except KeyboardInterrupt:
            return_code = -1
            failed_case = {
                "case": bench_case.json_dict(),
                "return_code": return_code,
                "error": "KeyboardInterrupt",
                "logs": {"stdout": "", "stderr": "KeyboardInterrupt"},
            }
            if not record_saved:
                save_benchmark_record(
                    record_path,
                    bench_case,
                    [],
                    failed_case,
                    hardware_hash,
                    software_hash,
                )
            failed_cases.append(failed_case)
            break
        except Exception as exc:
            return_code = -1
            failed_case = {
                "case": bench_case.json_dict(),
                "return_code": return_code,
                "error": repr(exc),
                "logs": {"stdout": "", "stderr": str(exc)},
            }
            if not record_saved:
                save_benchmark_record(
                    record_path,
                    bench_case,
                    [],
                    failed_case,
                    hardware_hash,
                    software_hash,
                )
            failed_cases.append(failed_case)
            logger.warning(f"Benchmark failed before subprocess execution: {exc}")
            if early_exit:
                break
    return return_code, results, failed_cases


def orchestrate_benchmarks(
    bench_cases: list[EstimatorCase],
    args,
) -> int:
    if args.log_level is not None:
        for log_type in ["runner", "bench"]:
            setattr(args, f"{log_type}_log_level", args.log_level)
    logger.setLevel(args.runner_log_level)

    env_info = get_environment_info()
    hardware_hash = get_hardware_hash(env_info["hardware"])
    software_hash = get_software_hash(env_info["software"])
    save_environment_sidecars(
        hardware_hash,
        software_hash,
        env_info,
        args.results_dir,
    )

    if args.prefetch_datasets:
        dataset_cases = {case.data.name(): case for case in bench_cases}
        n_datasets = len(dataset_cases)
        logger.debug(f"Unique dataset names to load:\n{list(dataset_cases.keys())}")
        n_proc = min([16, cpu_count(), n_datasets])
        logger.info(f"Prefetching {n_datasets} datasets with {n_proc} processes")
        with Pool(n_proc) as pool:
            pool.map(load_data_with_cleanup, dataset_cases.values())

    return_code, result, failed_cases = call_benchmarks(
        bench_cases,
        hardware_hash,
        software_hash,
        args.results_dir,
        args.bench_log_level,
        args.exit_on_error,
    )
    logger.debug(custom_format(result))
    logger.debug(custom_format(failed_cases))
    return return_code
