from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


JsonDict = dict[str, Any]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bench(_Section):
    n_runs: int = 10
    time_limit: float = 600
    taskset: str | int | None = None
    py_spy_profiling: bool = False
    flush_cache: bool = False
    gc_collect: bool = False
    cpu_profile: bool = False
    memory_profile: bool = False
    memory_profiling_interval: float = 0.001


class Algorithm(_Section):
    estimator: str
    estimator_params: JsonDict = Field(default_factory=dict)


class Data(_Section):
    source: str | None = None
    dataset: str | None = None
    id: int | str | None = None
    cache_directory: str | None = None
    raw_cache_directory: str | None = None
    generation_kwargs: JsonDict = Field(default_factory=dict)
    dataset_kwargs: JsonDict = Field(default_factory=dict)
    split_kwargs: JsonDict = Field(default_factory=dict)
    preprocessing_kwargs: JsonDict = Field(default_factory=dict)
    order: str | None = None
    dtype: str | None = None
    x_train: JsonDict | None = None
    x_test: JsonDict | None = None
    y_train: JsonDict | None = None
    y_test: JsonDict | None = None

    def name(self, shortened: bool = False) -> str:
        if self.dataset is not None:
            return self.dataset

        source = self.source
        generation_postfix = "".join(
            f"_{key}_{value}" for key, value in self.generation_kwargs.items()
        )
        dataset_postfix = "".join(
            f"_{key}_{value}" for key, value in self.dataset_kwargs.items()
        )

        if source == "fetch_openml":
            return f"openml_{self.id}"
        if source is not None and source.startswith("make_"):
            if shortened:
                return source.replace("classification", "clsf").replace(
                    "regression", "regr"
                )
            return f"{source}{generation_postfix}{dataset_postfix}"
        raise ValueError("Unable to get data name")


class Implementation(_Section):
    library: str
    device: str | None = None
    data_library: str | None = None
    sklearn_context: JsonDict | None = None
    sklearnex_context: JsonDict | None = None


class BenchCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bench: Bench = Field(default_factory=Bench)
    algorithm: Algorithm
    data: Data
    implementation: Implementation

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        name_args = [
            self.implementation.library,
            self.algorithm.estimator,
            self.data.name(shortened=shortened),
        ]
        if self.implementation.device is not None:
            name_args.append(self.implementation.device)
        return separator.join(name_args)


def _json_normalize(value: Any, context: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be JSON serializable: {exc}") from exc


def validate_case(case: dict) -> BenchCase:
    if not isinstance(case, dict):
        raise TypeError(f"case must be a dict, got {type(case).__name__}")
    normalized_input = _json_normalize(case, "case")
    try:
        validated = BenchCase.model_validate(normalized_input)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return validated


def _load_module_from_path(path: Path) -> ModuleType:
    module_name = "_sklbench_config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import config script: {path}")

    module = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent.resolve())
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(script_dir)
    return module


def load_cases_from_script(path: str | Path) -> list[BenchCase]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config script not found: {config_path}")
    if config_path.suffix != ".py":
        raise ValueError(f"Config must be a Python script: {config_path}")

    module = _load_module_from_path(config_path)
    generate_cases = getattr(module, "generate_cases", None)
    if generate_cases is None:
        raise ValueError(f"{config_path} does not define generate_cases()")
    if not callable(generate_cases):
        raise TypeError(f"{config_path}:generate_cases is not callable")

    raw_cases = generate_cases()
    if not isinstance(raw_cases, list):
        raise TypeError(
            f"{config_path}:generate_cases() must return a list of dicts, "
            f"got {type(raw_cases).__name__}"
        )

    cases = []
    for index, case in enumerate(raw_cases):
        try:
            cases.append(validate_case(case))
        except Exception as exc:
            raise ValueError(f"Invalid case at index {index}: {exc}") from exc
    return cases
