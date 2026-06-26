import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..benchmarks.sklearn_estimator import (
    estimator_to_task,
    get_context,
    get_estimator,
    get_subset_metrics_of_estimator,
)
from ..config import BenchCase, Bench
from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.logger import logger
from ..utils.measurement import measure_perf


def _as_jsonable(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _as_jsonable(inner_value)
            for key, inner_value in value.items()
            if _as_jsonable(inner_value) is not None
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [_as_jsonable(inner_value) for inner_value in value]
        return {"length": len(value)}
    return None


def _collect_model_attributes(estimator_instance) -> Dict[str, Any]:
    attributes = {}
    for attribute_name in [
        "n_iter_",
        "n_features_in_",
        "n_outputs_",
        "n_clusters_",
        "n_components_",
        "classes_",
        "support_vectors_",
    ]:
        if not hasattr(estimator_instance, attribute_name):
            continue
        value = getattr(estimator_instance, attribute_name)
        if attribute_name == "support_vectors_" and value is not None:
            value = len(value)
        jsonable = _as_jsonable(value)
        if jsonable is not None:
            attributes[attribute_name.rstrip("_")] = jsonable
    return attributes


def _measure_single_method(bench_params: Bench, method_instance, data_args):
    time_limit = bench_params.time_limit if bench_params.time_limit is not None else 3600
    cost_per_hour = bench_params.cost_per_hour if bench_params.cost_per_hour is not None else 0.0
    return measure_perf(
        method_instance,
        *data_args,
        n_runs=1,
        time_limit=time_limit,
        enable_itt=bench_params.vtune_profiling is not None,
        enable_cache_flushing=bench_params.flush_cache or False,
        enable_garbage_collection=bench_params.gc_collect or False,
        enable_cpu_profiling=bench_params.cpu_profile or False,
        enable_memory_profiling=bench_params.memory_profile or False,
        enable_nvml_profiling=False,
        cost_per_hour=cost_per_hour,
    )


def _split_time_and_metrics(result: Dict) -> Tuple[float, Dict]:
    time_values = result.get("time[ms]", [])
    time_value = time_values[0] if isinstance(time_values, list) else time_values
    return time_value, {key: value for key, value in result.items() if key != "time[ms]"}


def run_case_once(
    bench_case: BenchCase,
    estimator,
    task: str,
    data,
    data_description: Dict,
    repeat: int,
) -> Dict:
    X_train, X_test, y_train, y_test = data

    raw_metrics = {
        "fit": _measure_single_method(
            bench_case.bench,
            estimator.fit,
            (X_train,  y_train)
        ),
        "predict": _measure_single_method(
            bench_case,
            estimator.predict,
            (X_test, y_test),
        )
    }

    times = {}
    execution_metrics = {}
    for method, method_metrics in raw_metrics.items():
        times[method], execution_metrics[method] = _split_time_and_metrics(method_metrics)

    quality_metrics = {
        "fit": get_subset_metrics_of_estimator(task, "training", estimator, (X_train, y_train)),
        "predict": get_subset_metrics_of_estimator(task, "inference", estimator, (X_test, y_test))
    }

    data_desc = {
        "fit": dict(data_description["x_train"]),
        "predict": dict(data_description["x_test"]),
    }
    if "n_classes" in data_description:
        data_desc["fit"].update({"n_classes": data_description["n_classes"]})
        data_desc["predict"].update({"n_classes": data_description["n_classes"]})

    return {
        "case": bench_case.model_dump(mode="json", exclude_none=True),
        "repeat": repeat,
        "data_desc": data_desc,
        "time_ms": times,
        "metrics": quality_metrics,
        "execution_metrics": execution_metrics,
        "attributes": {_collect_model_attributes(estimator)},
    }


def run_case_to_jsonl(bench_case: BenchCase, output_jsonl: Path):
    bench_case_dict = bench_case.model_dump(mode="json", exclude_none=True)
    library_name = bench_case.implementation.library
    estimator_name = bench_case.algorithm.estimator
    estimator_class = get_estimator(library_name, estimator_name)
    task = estimator_to_task(estimator_name)

    raw_data, data_description = load_data(bench_case_dict)
    data, data_description = split_and_transform_data(
        bench_case_dict, raw_data, data_description
    )
    data = tuple(data)
    estimator_params = dict(bench_case.algorithm.estimator_params)
    n_runs = bench_case.bench.n_runs
    if n_runs is None:
        n_runs = 10

    context_class, context_params = get_context(bench_case_dict)
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for repeat in range(n_runs):
            with context_class(**context_params):
                row = run_case_once(
                    bench_case,
                    estimator_class(**estimator_params),
                    task,
                    data,
                    data_description,
                    repeat,
                )
            fp.write(json.dumps(row) + "\n")
            fp.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sklbench.runner")
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for benchmark runner",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.setLevel(args.log_level)
    with args.case_file.open("r", encoding="utf-8") as fp:
        bench_case = BenchCase.model_validate(json.load(fp))
    run_case_to_jsonl(bench_case, args.output_jsonl)
    return 0
