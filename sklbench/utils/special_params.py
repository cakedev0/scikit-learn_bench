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

from typing import Dict

import numpy as np
import pandas as pd
from psutil import cpu_count
from sklearn.metrics import euclidean_distances

from .bench_case import (
    get_bench_case_value,
    set_bench_case_value,
)
from .common import convert_to_numpy
from .custom_types import BenchCase
from .logger import logger

SP_VALUE_STR = "[SPECIAL_VALUE]"


def is_special_value(value) -> bool:
    return isinstance(value, str) and value.startswith(SP_VALUE_STR)


def get_ratio_from_n_jobs(n_jobs: str) -> float:
    args = n_jobs.split(":")
    if len(args) == 1:
        return 1.0
    elif len(args) == 2:
        return float(args[1])
    else:
        raise ValueError(f'Wrong arguments {args} in "n_jobs" special value')


def assign_device_case_values_on_run(bench_case: BenchCase):
    library = get_bench_case_value(bench_case, "implementation:library", None)
    estimator = get_bench_case_value(bench_case, "algorithm:estimator", None)
    device = get_bench_case_value(bench_case, "implementation:device", "default")
    sklearn_context = get_bench_case_value(
        bench_case, "implementation:sklearn_context", {}
    )
    sklearnex_context = get_bench_case_value(
        bench_case, "implementation:sklearnex_context", {}
    )
    if device != "default":
        if library == "xgboost" and estimator in ["XGBRegressor", "XGBClassifier"]:
            if device == "cpu" or any(map(device.startswith, ["gpu", "cuda"])):
                logger.debug(
                    f"Forwaring device '{device}' to XGBoost estimator parameters"
                )
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:device", device
                )
            else:
                raise ValueError(f"Unknown device '{device}' for xgboost {estimator}")
        elif library.startswith("sklearnex") or library.startswith("daal4py"):
            if device == "cpu":
                logger.debug(
                    "Skipping setting of 'target_offload' for CPU device "
                    "to avoid extra overheads"
                )
            elif (
                isinstance(sklearnex_context, dict)
                and sklearnex_context.get("array_api_dispatch", False)
            ) or (
                isinstance(sklearn_context, dict)
                and sklearn_context.get("array_api_dispatch", False)
            ):
                logger.debug(
                    f'Using device specification "{device}" for array API input arrays'
                )
            else:
                if not isinstance(sklearnex_context, dict):
                    sklearnex_context = {}
                sklearnex_context["target_offload"] = device
                set_bench_case_value(
                    bench_case,
                    "implementation:sklearnex_context",
                    sklearnex_context,
                )
        elif library == "sklbench.emulators.faiss" and estimator == "NearestNeighbors":
            set_bench_case_value(bench_case, "algorithm:estimator_params:device", device)
        elif library == "sklearn" and (
            isinstance(sklearn_context, dict)
            and sklearn_context.get("array_api_dispatch", False)
        ):
            logger.debug(
                f'Using device specification "{device}" for array API input arrays'
            )
        else:
            logger.warning(f'Device specification "{device}" is not used for this case')

    tree_method = get_bench_case_value(
        bench_case, "algorithm:estimator_params:tree_method", None
    )
    if tree_method == "gpu_hist":
        device = "gpu"
    set_bench_case_value(bench_case, "implementation:device", device)


def assign_case_special_values_from_data(
    bench_case: BenchCase, data, data_description: Dict
):
    # Note: data = (x_train, y_train, x_test, y_test)
    library = get_bench_case_value(bench_case, "implementation:library", None)
    estimator = get_bench_case_value(bench_case, "algorithm:estimator", None)

    scale_pos_weight = get_bench_case_value(
        bench_case, "algorithm:estimator_params:scale_pos_weight", None
    )
    if (
        is_special_value(scale_pos_weight)
        and scale_pos_weight.replace(SP_VALUE_STR, "") == "auto"
        and (library.endswith("gbm") or library.endswith("boost"))
        and estimator.endswith("Classifier")
    ):
        y_train = convert_to_numpy(data[1])
        value_counts = pd.Series(y_train).value_counts().sort_index()
        if len(value_counts) != 2:
            logger.info(
                f"Number of classes ({len(value_counts)}) != 2 "
                'while "scale_pos_weight" is set to "auto". '
                "This parameter is removed from estimator parameters."
            )
            set_bench_case_value(
                bench_case, "algorithm:estimator_params:scale_pos_weight", None
            )
        else:
            scale_pos_weight = value_counts.iloc[0] / value_counts.iloc[1]
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:scale_pos_weight",
                scale_pos_weight,
            )

    n_clusters = get_bench_case_value(
        bench_case, "algorithm:estimator_params:n_clusters", None
    )
    if is_special_value(n_clusters) and n_clusters.replace(SP_VALUE_STR, "") == "auto":
        n_clusters = data_description.get("n_clusters", None)
        n_classes = data_description.get("n_classes", None)
        n_clusters_per_class = data_description.get("n_clusters_per_class", 1)
        if n_clusters is not None:
            if isinstance(n_clusters, int):
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:n_clusters", n_clusters
                )
            else:
                raise ValueError(
                    f"n_clusters={n_clusters} of type {type(n_clusters)} "
                    "from data description is not integer."
                )
        elif n_classes is not None:
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:n_clusters",
                n_classes * n_clusters_per_class,
            )
        else:
            raise ValueError(
                "Unable to auto-assign n_clusters: "
                "data description doesn't have n_clusters or n_classes"
            )

    eps = get_bench_case_value(bench_case, "algorithm:estimator_params:eps", None)
    if is_special_value(eps) and eps.replace(SP_VALUE_STR, "").startswith(
        "distances_quantile"
    ):
        x_train = convert_to_numpy(data[0])
        quantile = float(eps.replace(SP_VALUE_STR, "").split(":")[1])
        subsample = list(getattr(x_train, "index", np.arange(x_train.shape[0])))
        np.random.seed(42)
        np.random.shuffle(subsample)
        subsample = subsample[: min(x_train.shape[0], 1000)]
        x_sample = (
            x_train.loc[subsample] if hasattr(x_train, "loc") else x_train[subsample]
        )
        x_sample = x_sample.astype("float32")
        dist = np.tril(euclidean_distances(x_sample, x_sample)).reshape(-1)
        dist = dist[dist != 0]
        quantile = float(np.quantile(dist, quantile))
        set_bench_case_value(bench_case, "algorithm:estimator_params:eps", quantile)


def assign_case_special_values_on_run(
    bench_case: BenchCase, data, data_description: Dict
):
    # Note: data = (x_train, y_train, x_test, y_train)
    library = get_bench_case_value(bench_case, "implementation:library", None)
    estimator = get_bench_case_value(bench_case, "algorithm:estimator", None)
    # device-related parameters assignment
    device = get_bench_case_value(bench_case, "implementation:device", "default")
    sklearn_context = get_bench_case_value(
        bench_case, "implementation:sklearn_context", {}
    )
    sklearnex_context = get_bench_case_value(
        bench_case, "implementation:sklearnex_context", {}
    )
    if device != "default":
        # xgboost tree method assignment branch
        if library == "xgboost" and estimator in ["XGBRegressor", "XGBClassifier"]:
            if device == "cpu" or any(map(device.startswith, ["gpu", "cuda"])):
                logger.debug(
                    f"Forwaring device '{device}' to XGBoost estimator parameters"
                )
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:device", device
                )
            else:
                raise ValueError(f"Unknown device '{device}' for xgboost {estimator}")
        # set target offload for execution context
        elif library.startswith("sklearnex") or library.startswith("daal4py"):
            if device == "cpu":
                logger.debug(
                    "Skipping setting of 'target_offload' for CPU device "
                    "to avoid extra overheads"
                )
            elif (
                isinstance(sklearnex_context, dict)
                and sklearnex_context.get("array_api_dispatch", False)
            ) or (
                isinstance(sklearn_context, dict)
                and sklearn_context.get("array_api_dispatch", False)
            ):
                logger.debug(
                    f'Using device specification "{device}" for array API input arrays'
                )
            else:
                if not isinstance(sklearnex_context, dict):
                    sklearnex_context = {}
                sklearnex_context["target_offload"] = device
                set_bench_case_value(
                    bench_case,
                    "implementation:sklearnex_context",
                    sklearnex_context,
                )
        # faiss GPU algorithm selection
        elif library == "sklbench.emulators.faiss" and estimator == "NearestNeighbors":
            set_bench_case_value(bench_case, "algorithm:estimator_params:device", device)
        elif library == "sklearn" and (
            isinstance(sklearn_context, dict)
            and sklearn_context.get("array_api_dispatch", False)
        ):
            logger.debug(
                f'Using device specification "{device}" for array API input arrays'
            )
        else:
            logger.warning(f'Device specification "{device}" is not used for this case')
    # assign "default" or changed device for output
    tree_method = get_bench_case_value(
        bench_case, "algorithm:estimator_params:tree_method", None
    )
    if tree_method == "gpu_hist":
        device = "gpu"
    set_bench_case_value(bench_case, "implementation:device", device)
    # n_jobs
    n_jobs = get_bench_case_value(bench_case, "algorithm:estimator_params:n_jobs", None)
    if is_special_value(n_jobs):
        n_jobs = n_jobs.replace(SP_VALUE_STR, "")
        if n_jobs.startswith("physical_cpus"):
            n_cpus = cpu_count(logical=False)
        elif n_jobs.startswith("logical_cpus"):
            n_cpus = cpu_count(logical=True)
        else:
            raise ValueError(f'Unknown special value {n_jobs} for "n_jobs"')
        n_jobs = int(n_cpus * get_ratio_from_n_jobs(n_jobs))
        set_bench_case_value(bench_case, "algorithm:estimator_params:n_jobs", n_jobs)
    # classes balance for GBT frameworks
    scale_pos_weight = get_bench_case_value(
        bench_case, "algorithm:estimator_params:scale_pos_weight", None
    )
    if (
        is_special_value(scale_pos_weight)
        and scale_pos_weight.replace(SP_VALUE_STR, "") == "auto"
        and (library.endswith("gbm") or library.endswith("boost"))
        and estimator.endswith("Classifier")
    ):
        y_train = convert_to_numpy(data[1])
        value_counts = pd.Series(y_train).value_counts().sort_index()
        if len(value_counts) != 2:
            logger.info(
                f"Number of classes ({len(value_counts)}) != 2 "
                'while "scale_pos_weight" is set to "auto". '
                "This parameter is removed from estimator parameters."
            )
            set_bench_case_value(
                bench_case, "algorithm:estimator_params:scale_pos_weight", None
            )
        else:
            scale_pos_weight = value_counts.iloc[0] / value_counts.iloc[1]
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:scale_pos_weight",
                scale_pos_weight,
            )
    # number of classes assignment for multiclass LightGBM
    num_classes = get_bench_case_value(
        bench_case, "algorithm:estimator_params:num_classes", None
    )
    if is_special_value(num_classes) and num_classes.replace(SP_VALUE_STR, "") == "auto":
        set_bench_case_value(
            bench_case,
            "algorithm:estimator_params:num_classes",
            data_description.get("n_classes", None),
        )
    # "n_clusters" auto assignment from data description
    n_clusters = get_bench_case_value(
        bench_case, "algorithm:estimator_params:n_clusters", None
    )
    if is_special_value(n_clusters) and n_clusters.replace(SP_VALUE_STR, "") == "auto":
        n_clusters = data_description.get("n_clusters", None)
        n_classes = data_description.get("n_classes", None)
        n_clusters_per_class = data_description.get("n_clusters_per_class", 1)
        if n_clusters is not None:
            if isinstance(n_clusters, int):
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:n_clusters", n_clusters
                )
            else:
                raise ValueError(
                    f"n_clusters={n_clusters} of type {type(n_clusters)} "
                    "from data description is not integer."
                )
        elif n_classes is not None:
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:n_clusters",
                n_classes * n_clusters_per_class,
            )
        else:
            raise ValueError(
                "Unable to auto-assign n_clusters: "
                "data description doesn't have n_clusters or n_classes"
            )
    # "eps" auto assignment for DBSCAN
    eps = get_bench_case_value(bench_case, "algorithm:estimator_params:eps", None)
    if is_special_value(eps) and eps.replace(SP_VALUE_STR, "").startswith(
        "distances_quantile"
    ):
        x_train = convert_to_numpy(data[0])
        quantile = float(eps.replace(SP_VALUE_STR, "").split(":")[1])
        # subsample of x_train is used to avoid reaching of memory limit for large matrices
        subsample = list(getattr(x_train, "index", np.arange(x_train.shape[0])))
        np.random.seed(42)
        np.random.shuffle(subsample)
        subsample = subsample[: min(x_train.shape[0], 1000)]
        x_sample = (
            x_train.loc[subsample] if hasattr(x_train, "loc") else x_train[subsample]
        )
        # conversion to lower precision is required
        # to produce same distances quantile for different dtypes of x
        x_sample = x_sample.astype("float32")
        dist = np.tril(euclidean_distances(x_sample, x_sample)).reshape(-1)
        dist = dist[dist != 0]
        quantile = float(np.quantile(dist, quantile))
        set_bench_case_value(bench_case, "algorithm:estimator_params:eps", quantile)
