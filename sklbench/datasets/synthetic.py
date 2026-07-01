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
from collections.abc import Sequence

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state


ColumnSpec = str | Sequence[str]


def transform_columns(
    x: np.ndarray, columns: ColumnSpec, rng: np.random.RandomState
) -> None:
    n_features = x.shape[1]
    if columns == "mix":
        columns = ["continuous", "binary", "long-tail"] * n_features
        columns = columns[:n_features]
    elif isinstance(columns, str):
        columns = [columns] * n_features

    for col_idx in rng.permutation(n_features):
        col_type = columns[col_idx]
        values = x[:, col_idx]
        if col_type == "continuous":
            continue
        if col_type == "binary":
            thresholds = np.quantile(values, np.sort(rng.uniform(size=2)))
            x[:, col_idx] = np.searchsorted(thresholds, values) == 1
        elif col_type == "long-tail":
            noise_ratio = rng.uniform(0.05, 0.5)
            n_bins = max(1, min(3, rng.poisson(10)))
            binned = np.searchsorted(
                np.linspace(values.min() + 1e-10, values.max(), num=n_bins),
                values,
            )
            mask = rng.rand(binned.size) < noise_ratio
            if mask.any():
                binned[mask] = rng.geometric(min(1, 20 / mask.sum()), size=mask.sum())
            x[:, col_idx] = binned
        else:
            raise ValueError(col_type)


def make_trees_regression_data(*, columns: ColumnSpec, random_state=None, **kwargs):
    rng = check_random_state(random_state)
    x, y = make_regression(**kwargs, random_state=rng)
    transform_columns(x, columns, rng)
    return x, y


def make_trees_classification_data(*, columns: ColumnSpec, random_state=None, **kwargs):
    rng = check_random_state(random_state)
    x, y = make_classification(**kwargs, random_state=rng)
    transform_columns(x, columns, rng)
    return x, y
