import json
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from psutil import cpu_count
from tqdm import tqdm

from ..config import BenchCase
from ..datasets import load_data_with_cleanup
from ..common.filtering import bench_case_filter
from ..utils.common import custom_format, hash_from_json_repr
from ..utils.logger import logger
from .commands import run_runner_from_case
from .env import get_environment_info


def get_hardware_hash(hardware_info: Dict) -> str:
    return hash_from_json_repr(hardware_info, hash_limit=6)


def get_software_hash(software_info: Dict) -> str:
    return hash_from_json_repr(software_info, hash_limit=6)


def _append_metric_value(values: List, value):
    if isinstance(value, list):
        values.extend(value)
    else:
        values.append(value)


def _merge_method_metrics(rows: List[Dict]) -> Dict:
    first_row = rows[0]
    metrics = {
        method: dict(method_metrics)
        for method, method_metrics in first_row.get("metrics", {}).items()
    }
    for row in rows:
        for method, method_metrics in row.get("execution_metrics", {}).items():
            method_result = metrics.setdefault(method, {})
            for metric_name, metric_value in method_metrics.items():
                _append_metric_value(
                    method_result.setdefault(metric_name, []), metric_value
                )

    attributes = first_row.get("attributes", {})
    if attributes:
        metrics.setdefault("fit", {}).update(attributes)
    return metrics


def aggregate_runner_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "data_desc": {},
            "time[ms]": {},
            "metrics": {},
            "logs": {"stdout": "", "stderr": ""},
        }

    method_names = []
    for row in rows:
        for method in row.get("time_ms", {}):
            if method not in method_names:
                method_names.append(method)

    times = {method: [] for method in method_names}
    for row in rows:
        for method in method_names:
            if method in row.get("time_ms", {}):
                times[method].append(row["time_ms"][method])

    first_row = rows[0]
    return {
        "data_desc": first_row.get("data_desc", {}),
        "time[ms]": times,
        "metrics": _merge_method_metrics(rows),
        "logs": first_row.get("logs", {"stdout": "", "stderr": ""}),
    }


def call_benchmarks(
    bench_cases: List[BenchCase],
    filters: List[BenchCase],
    log_level: str = "WARNING",
    early_exit: bool = False,
) -> Tuple[int, List[Dict], List[Dict]]:
    results = []
    failed_cases = []
    return_code = 0
    filtered_cases = [
        bench_case for bench_case in bench_cases if bench_case_filter(bench_case, filters)
    ]
    if len(filtered_cases) != len(bench_cases):
        logger.info(
            "Filtering reduced number of cases from "
            f"{len(bench_cases)} to {len(filtered_cases)}."
        )

    bench_cases_with_pbar = tqdm(filtered_cases)
    for bench_case in bench_cases_with_pbar:
        bench_cases_with_pbar.set_description(
            custom_format(
                bench_case.name(shortened=True), bcolor="HEADER"
            )
        )
        try:
            bench_return_code, rows, failed_case = run_runner_from_case(
                bench_case, log_level
            )
            if bench_return_code != 0:
                return_code = bench_return_code
                if failed_case is not None:
                    failed_cases.append(failed_case)
                if early_exit:
                    break
            results.append(
                {"case": bench_case.json_dict(), **aggregate_runner_rows(rows)}
            )
        except KeyboardInterrupt:
            return_code = -1
            break
        except Exception as exc:
            return_code = -1
            failed_cases.append(
                {
                    "case": bench_case.json_dict(),
                    "return_code": return_code,
                    "error": repr(exc),
                    "logs": {"stdout": "", "stderr": str(exc)},
                }
            )
            logger.warning(f"Benchmark failed before subprocess execution: {exc}")
            if early_exit:
                break
    return return_code, results, failed_cases


def save_results(
    benchmark_results: List[Dict],
    failed_cases: List[Dict],
    hardware_hash: str,
    software_hash: str,
    env_info: Dict,
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

    result = {
        "hardware_hash": hardware_hash,
        "software_hash": software_hash,
        "bench_cases": benchmark_results,
        "failed_cases": failed_cases,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_file = results_root / f"{timestamp}.json"
    with result_file.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=4)
    logger.warning(f"Benchmark results saved to {result_file}")


def orchestrate_benchmarks(
    bench_cases: List[BenchCase],
    filters: List[BenchCase],
    args,
) -> int:
    if args.log_level is not None:
        for log_type in ["runner", "bench"]:
            setattr(args, f"{log_type}_log_level", args.log_level)
    logger.setLevel(args.runner_log_level)

    env_info = get_environment_info()
    hardware_hash = get_hardware_hash(env_info["hardware"])
    software_hash = get_software_hash(env_info["software"])

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
        filters,
        args.bench_log_level,
        args.exit_on_error,
    )
    logger.debug(custom_format(result))
    save_results(
        result,
        failed_cases,
        hardware_hash,
        software_hash,
        env_info,
        args.results_dir,
    )
    return return_code
