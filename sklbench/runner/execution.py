import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..benchmarks.sklearn_estimator import (
    create_online_function,
    estimator_to_task,
    get_context,
    get_estimator,
    get_estimator_methods,
    get_subset_metrics_of_estimator,
    validate_estimator_params,
)
from ..config import validate_case
from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.bench_case import get_bench_case_value
from ..utils.common import custom_format
from ..utils.custom_types import BenchCase
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


def _inject_num_classes_if_needed(
    bench_case: BenchCase,
    estimator_class,
    estimator_params: Dict,
    data_description: Dict,
) -> Dict:
    if "num_classes" in estimator_params:
        return estimator_params

    n_classes = data_description.get("n_classes")
    if not isinstance(n_classes, int) or n_classes <= 2:
        return estimator_params

    library = get_bench_case_value(bench_case, "implementation:library", "")
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator", "")
    objective = estimator_params.get("objective")
    is_lightgbm_multiclass = (
        library == "lightgbm"
        and estimator_name == "LGBMClassifier"
        and isinstance(objective, str)
        and objective.startswith("multiclass")
    )
    try:
        signature = inspect.signature(estimator_class.__init__)
        accepts_num_classes = "num_classes" in signature.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_num_classes = False
        accepts_kwargs = False

    if accepts_num_classes or (is_lightgbm_multiclass and accepts_kwargs):
        estimator_params = dict(estimator_params)
        estimator_params["num_classes"] = n_classes
    return estimator_params


def _method_data_args(method_instance, stage, x_train, x_test, y_train, y_test):
    if "y" in list(inspect.signature(method_instance).parameters):
        return (x_train, y_train) if stage == "training" else (x_test, y_test)
    return (x_train,) if stage == "training" else (x_test,)


def _maybe_modelbuilder_method(bench_case, estimator_instance, method, method_instance):
    enable_modelbuilders = get_bench_case_value(
        bench_case, "algorithm:enable_modelbuilders", False
    )
    if not enable_modelbuilders:
        return method_instance

    import daal4py

    if hasattr(estimator_instance, "get_booster"):
        daal_model = daal4py.mb.convert_model(estimator_instance.get_booster())
    elif hasattr(estimator_instance, "booster_"):
        daal_model = daal4py.mb.convert_model(estimator_instance.booster_)
    else:
        raise ValueError("Unable to convert model to daal4py GBT format.")
    return getattr(daal_model, method)


def _measure_single_method(bench_case, method_instance, data_args):
    return measure_perf(
        method_instance,
        *data_args,
        n_runs=1,
        time_limit=get_bench_case_value(bench_case, "bench:time_limit", 3600),
        enable_itt=get_bench_case_value(bench_case, "bench:vtune_profiling") is not None,
        enable_cache_flushing=get_bench_case_value(
            bench_case, "bench:flush_cache", False
        ),
        enable_garbage_collection=get_bench_case_value(
            bench_case, "bench:gc_collect", False
        ),
        enable_cpu_profiling=get_bench_case_value(bench_case, "bench:cpu_profile", False),
        enable_memory_profiling=get_bench_case_value(
            bench_case, "bench:memory_profile", False
        ),
        enable_nvml_profiling=False,
        cost_per_hour=get_bench_case_value(bench_case, "bench:cost_per_hour", 0.0),
    )


def _split_time_and_metrics(result: Dict) -> Tuple[float, Dict]:
    time_values = result.get("time[ms]", [])
    time_value = time_values[0] if isinstance(time_values, list) else time_values
    return time_value, {key: value for key, value in result.items() if key != "time[ms]"}


def run_case_once(
    bench_case: BenchCase,
    estimator_class,
    estimator_params: Dict,
    estimator_methods: Dict,
    task: str,
    data,
    data_description: Dict,
    repeat: int,
) -> Dict:
    x_train, x_test, y_train, y_test = data
    estimator_instance = estimator_class(**estimator_params)
    raw_metrics = {}
    for stage, methods in estimator_methods.items():
        for method in methods:
            if not hasattr(estimator_instance, method):
                continue
            method_instance = getattr(estimator_instance, method)
            data_args = _method_data_args(
                method_instance, stage, x_train, x_test, y_train, y_test
            )
            batch_size = get_bench_case_value(
                bench_case, f"algorithm:batch_size:{stage}"
            )
            if batch_size is not None:
                method_instance = create_online_function(
                    method_instance, data_args, batch_size
                )
            if stage == "inference":
                method_instance = _maybe_modelbuilder_method(
                    bench_case, estimator_instance, method, method_instance
                )

            raw_metrics[method] = _measure_single_method(
                bench_case, method_instance, data_args
            )

    quality_metrics = {
        "training": get_subset_metrics_of_estimator(
            task, "training", estimator_instance, (x_train, y_train)
        ),
        "inference": get_subset_metrics_of_estimator(
            task, "inference", estimator_instance, (x_test, y_test)
        ),
    }

    times = {}
    metrics = {}
    for method, method_metrics in raw_metrics.items():
        method_time, method_extra_metrics = _split_time_and_metrics(method_metrics)
        times[method] = method_time
        for stage, methods in estimator_methods.items():
            if method in methods:
                method_extra_metrics.update(quality_metrics[stage])
        metrics[method] = method_extra_metrics

    data_desc = {
        "fit": dict(data_description["x_train"]),
        "predict": dict(data_description["x_test"]),
    }
    for stage in estimator_methods.keys():
        output_stage = "fit" if stage == "training" else "predict"
        data_desc[output_stage].update(
            {
                "batch_size": get_bench_case_value(
                    bench_case, f"algorithm:batch_size:{stage}"
                )
            }
        )
        if "n_classes" in data_description:
            data_desc[output_stage].update({"n_classes": data_description["n_classes"]})

    return {
        "case": bench_case,
        "repeat": repeat,
        "data_desc": data_desc,
        "time_ms": times,
        "metrics": metrics,
        "attributes": {"fit": _collect_model_attributes(estimator_instance)},
        "warnings": [],
    }


def run_case_to_jsonl(bench_case: BenchCase, output_jsonl: Path):
    library_name = get_bench_case_value(bench_case, "implementation:library")
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator")
    estimator_class = get_estimator(library_name, estimator_name)
    task = estimator_to_task(estimator_name)

    raw_data, data_description = load_data(bench_case)
    data, data_description = split_and_transform_data(
        bench_case, raw_data, data_description
    )
    data = tuple(data)
    estimator_params = get_bench_case_value(
        bench_case, "algorithm:estimator_params", dict()
    )
    estimator_params = _inject_num_classes_if_needed(
        bench_case, estimator_class, estimator_params, data_description
    )
    estimator_params = validate_estimator_params(estimator_class, estimator_params)
    logger.debug(f"Estimator parameters:\n{custom_format(estimator_params)}")
    estimator_methods = get_estimator_methods(bench_case)
    n_runs = get_bench_case_value(bench_case, "bench:n_runs", 10)

    context_class, context_params = get_context(bench_case)
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for repeat in range(n_runs):
            with context_class(**context_params):
                row = run_case_once(
                    bench_case,
                    estimator_class,
                    estimator_params,
                    estimator_methods,
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
        bench_case = validate_case(json.load(fp))
    run_case_to_jsonl(bench_case, args.output_jsonl)
    return 0
