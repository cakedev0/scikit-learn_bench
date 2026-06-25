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
    model_config = ConfigDict(extra="allow")


class Bench(_Section):
    n_runs: int | None = None
    time_limit: float | None = None
    taskset: str | int | None = None
    distributor: str | None = None
    mpi_params: JsonDict | None = None
    vtune_profiling: str | None = None
    vtune_results_directory: str | None = None
    flush_cache: bool | None = None
    gc_collect: bool | None = None
    cpu_profile: bool | None = None
    memory_profile: bool | None = None
    cost_per_hour: float | None = None


class Algorithm(_Section):
    estimator: str | None = None
    function: str | None = None
    estimator_params: JsonDict = Field(default_factory=dict)
    estimator_methods: JsonDict | None = None
    batch_size: JsonDict | None = None
    enable_modelbuilders: bool | None = None


class Data(_Section):
    source: str | None = None
    dataset: str | None = None
    id: int | str | None = None
    generation_kwargs: JsonDict = Field(default_factory=dict)
    dataset_kwargs: JsonDict = Field(default_factory=dict)
    split_kwargs: JsonDict = Field(default_factory=dict)
    preprocessing_kwargs: JsonDict = Field(default_factory=dict)
    order: str | None = None


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


def _json_normalize(value: Any, context: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be JSON serializable: {exc}") from exc


def validate_case(case: dict) -> dict:
    """Validate and normalize one benchmark case.

    The returned value contains only JSON-serializable Python objects and omits
    fields whose value is ``None``.
    """

    if not isinstance(case, dict):
        raise TypeError(f"case must be a dict, got {type(case).__name__}")
    normalized_input = _json_normalize(case, "case")
    try:
        validated = BenchCase.model_validate(normalized_input)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    normalized = validated.model_dump(mode="json", exclude_none=True)
    return normalized


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


def load_cases_from_script(path: str | Path) -> list[dict]:
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
