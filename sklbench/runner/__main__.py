import argparse
import inspect
import json
import statistics
import sys
import timeit
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..config import BenchCase
from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.logger import logger
from .estimator import estimator_to_task, get_context, get_estimator
from .measurement import measure_perf
from .metrics import get_subset_metrics_of_estimator


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


def _collect_model_attributes(estimator) -> Dict[str, Any]:
    attributes = {}
    for attribute_name in [
        "n_iter_",
        "solver_",
        "n_features_in_",
        "n_outputs_",
        "n_clusters_",
        "n_components_",
        "oob_score_",
    ]:
        if not hasattr(estimator, attribute_name):
            continue
        value = getattr(estimator, attribute_name)
        if attribute_name == "support_vectors_" and value is not None:
            value = len(value)
        jsonable = _as_jsonable(value)
        if jsonable is not None:
            attributes[attribute_name.rstrip("_")] = jsonable

    if hasattr(estimator, "estimators_") and hasattr(estimator.estimators_[0], "tree_"):
        attributes["avg_n_leaves"] = statistics.mean(
            tree.tree_.n_leaves
            for tree in estimator.estimators_
        )
        attributes["avg_max_depth"] = statistics.mean(
            tree.tree_.max_depth
            for tree in estimator.estimators_
        )

    for name, value in vars(estimator).items():
        if not name.endswith("_") or name.startswith("_"):
            # not a public attribute
            continue
        attribute_name = name.rstrip("_")
        if attribute_name.rstrip("_") in attributes:
            # already collected
            continue
        if isinstance(value, (int, float, str)):
            attributes[attribute_name] = _as_jsonable(value)

    return attributes


def run_case_once(
    bench_case: BenchCase,
    estimator,
    data,
    data_description: Dict,
) -> Dict:
    task = estimator_to_task(bench_case.algorithm.estimator)
    X_train, X_test, y_train, y_test = data

    times = {}
    profiling_metrics = {}

    times["fit"], profiling_metrics["fit"] = measure_perf(
        estimator.fit,
        X_train,
        y_train,
        bench_params=bench_case.bench,
    )

    times["predict"], profiling_metrics["predict"] = measure_perf(
        estimator.predict,
        X_test,
        bench_params=bench_case.bench,
    )

    quality_metrics = {
        "fit": get_subset_metrics_of_estimator(
            task, "training", estimator, (X_train, y_train)
        ),
        "predict": get_subset_metrics_of_estimator(
            task, "inference", estimator, (X_test, y_test)
        ),
    }

    data_desc = {
        "fit": dict(data_description["x_train"]),
        "predict": dict(data_description["x_test"]),
    }
    if "n_classes" in data_description:
        data_desc["fit"].update({"n_classes": data_description["n_classes"]})
        data_desc["predict"].update({"n_classes": data_description["n_classes"]})

    return {
        "data_desc": data_desc,
        "time_ms": times,
        "metrics": quality_metrics,
        "profiling_metrics": profiling_metrics,
        "attributes": _collect_model_attributes(estimator),
    }


def estimator_params_for_repeat(
    estimator_class, estimator_params: Dict, repeat: int
) -> Dict:
    params = dict(estimator_params)
    if "random_state" in params:
        return params

    try:
        signature = inspect.signature(estimator_class)
    except (TypeError, ValueError):
        return params
    if "random_state" in signature.parameters:
        params["random_state"] = repeat
    return params


def run_case_to_jsonl(bench_case: BenchCase, output_jsonl: Path):
    library_name = bench_case.implementation.library
    estimator_name = bench_case.algorithm.estimator
    estimator_class = get_estimator(library_name, estimator_name)

    raw_data, data_description = load_data(bench_case)
    data, data_description = split_and_transform_data(
        bench_case, raw_data, data_description
    )
    data = tuple(data)
    estimator_params = dict(bench_case.algorithm.estimator_params)
    n_runs = bench_case.bench.n_runs
    time_limit = bench_case.bench.time_limit

    with output_jsonl.open("w", encoding="utf-8") as fp, get_context(
        bench_case.implementation
    ):
        t0 = timeit.default_timer()
        for repeat in range(n_runs):
            repeat_estimator_params = estimator_params_for_repeat(
                estimator_class, estimator_params, repeat
            )
            row = run_case_once(
                bench_case,
                estimator_class(**repeat_estimator_params),
                data,
                data_description,
            )
            fp.write(json.dumps(row, default=_as_jsonable) + "\n")
            fp.flush()
            if timeit.default_timer() - t0 > time_limit:
                logger.warning(f"runner exceeded time limit ({time_limit} seconds)")
                break


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


if __name__ == "__main__":
    sys.exit(main())
