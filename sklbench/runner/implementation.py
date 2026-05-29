# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================


import argparse
import json
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from psutil import cpu_count
from tqdm import tqdm

from ..datasets import load_data_with_cleanup
from ..utils.bench_case import get_bench_case_name, get_data_name
from ..utils.common import custom_format, hash_from_json_repr
from ..utils.config import early_filtering, generate_bench_cases, generate_bench_filters
from ..utils.custom_types import BenchCase
from ..utils.env import get_environment_info
from ..utils.logger import logger
from .commands_helper import run_benchmark_from_case


def call_benchmarks(
    bench_cases: List[BenchCase],
    filters: List[BenchCase],
    log_level: str = "WARNING",
    early_exit: bool = False,
) -> Tuple[int, List[Dict]]:
    """Iterates over benchmarking cases with progress bar and combines their results"""
    results = list()
    return_code = 0
    bench_cases_with_pbar = tqdm(bench_cases)
    for bench_case in bench_cases_with_pbar:
        bench_cases_with_pbar.set_description(
            custom_format(
                get_bench_case_name(bench_case, shortened=True), bcolor="HEADER"
            )
        )
        try:
            bench_return_code, bench_entries = run_benchmark_from_case(
                bench_case, filters, log_level
            )
            if bench_return_code != 0:
                return_code = bench_return_code
                if early_exit:
                    break
            results.extend(bench_entries)
        except KeyboardInterrupt:
            return_code = -1
            break
    return return_code, results


def save_results(bench_cases: List[Dict], env_name: str, env_info: Dict, results_dir: str):
    results_root = Path(results_dir)
    env_dir = results_root / "envs"
    bench_results_dir = results_root / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    bench_results_dir.mkdir(parents=True, exist_ok=True)

    env_file = env_dir / f"{env_name}.json"
    try:
        with open(env_file, "x") as fp:
            json.dump(env_info, fp, indent=4)
    except FileExistsError:
        pass

    result = {"bench_cases": bench_cases}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_file = bench_results_dir / f"results_{timestamp}.json"
    with open(result_file, "x") as fp:
        json.dump(result, fp, indent=4)
    logger.warning(f"Benchmark results saved to {result_file}")


def run_benchmarks(args: argparse.Namespace) -> int:
    # overwrite all logging levels if requested
    if args.log_level is not None:
        for log_type in ["runner", "bench"]:
            setattr(args, f"{log_type}_log_level", args.log_level)
    # set logging level
    logger.setLevel(args.runner_log_level)

    env_info = get_environment_info()
    env_name = hash_from_json_repr(env_info, hash_limit=6)

    # find and parse configs
    bench_cases = generate_bench_cases(args)

    # get parameter filters
    param_filters = generate_bench_filters(args.parameter_filters)

    # perform early filtering based on 'data' parameters and
    # some of 'algorithm' parameters assuming they were already assigned
    bench_cases = early_filtering(bench_cases, param_filters)

    # prefetch datasets
    if args.prefetch_datasets:
        # trick: get unique dataset names only to avoid loading of same dataset
        # by different cases/processes
        dataset_cases = {get_data_name(case): case for case in bench_cases}
        n_datasets = len(dataset_cases)
        logger.debug(f"Unique dataset names to load:\n{list(dataset_cases.keys())}")
        n_proc = min([16, cpu_count(), n_datasets])
        logger.info(f"Prefetching {n_datasets} datasets with {n_proc} processes")
        with Pool(n_proc) as pool:
            pool.map(load_data_with_cleanup, dataset_cases.values())

    # run bench_cases
    return_code, result = call_benchmarks(
        bench_cases,
        param_filters,
        args.bench_log_level,
        args.exit_on_error,
    )

    # output raw result
    logger.debug(custom_format(result))

    # save results to append-only results directory
    save_results(result, env_name, env_info, args.results_dir)

    return return_code
