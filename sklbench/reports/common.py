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

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


RESULT_FILE_RE = re.compile(r"^(?:.+_)?(\d{8}T\d{6}(?:\d{6})?Z)\.json$")


@dataclass(frozen=True)
class ResultFile:
    path: Path
    env_name: str
    timestamp: datetime
    environment: Dict[str, Any]
    bench_cases: List[Dict[str, Any]]


def parse_result_timestamp(path: Path) -> datetime:
    match = RESULT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(
            f"Result filename '{path}' does not end with _<datetime>.json"
        )
    timestamp = match.group(1)
    date_format = "%Y%m%dT%H%M%S%fZ" if len(timestamp) == 22 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(timestamp, date_format).replace(tzinfo=timezone.utc)


def load_result_file(path: Path) -> ResultFile:
    if not path.is_file():
        raise FileNotFoundError(f"Result input must be a file: {path}")

    env_root = path.parent.parent.parent
    hardware_env_name = path.parent.parent.name
    software_env_name = path.parent.name
    hardware_env_file = env_root / "hardware-envs" / f"{hardware_env_name}.json"
    software_env_file = env_root / "software-envs" / f"{software_env_name}.json"

    if hardware_env_file.is_file() and software_env_file.is_file():
        env_name = f"{hardware_env_name}/{software_env_name}"
        with open(hardware_env_file, "r") as fp:
            hardware = json.load(fp)
        with open(software_env_file, "r") as fp:
            software = json.load(fp)
        environment = {"hardware": hardware, "software": software}
    else:
        env_name = path.parent.name
        env_file = path.parent.parent / "envs" / f"{env_name}.json"
        if not env_file.is_file():
            raise FileNotFoundError(
                f"Unable to find environment files for '{path}': expected either "
                f"{hardware_env_file} and {software_env_file}, or {env_file}"
            )
        with open(env_file, "r") as fp:
            environment = json.load(fp)

    with open(path, "r") as fp:
        result = json.load(fp)

    if "bench_cases" not in result:
        raise ValueError(f"Result file '{path}' does not contain 'bench_cases'")

    return ResultFile(
        path=path,
        env_name=env_name,
        timestamp=parse_result_timestamp(path),
        environment=environment,
        bench_cases=result["bench_cases"],
    )


def load_result_files(paths: Iterable[Path]) -> List[ResultFile]:
    return [load_result_file(path) for path in paths]


def without_keys(value: Any, excluded_names: set, excluded_prefixes: tuple = ()) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, nested_value in value.items():
            if key in excluded_names or any(
                key.startswith(prefix) for prefix in excluded_prefixes
            ):
                continue
            result[key] = without_keys(nested_value, excluded_names, excluded_prefixes)
        return result
    if isinstance(value, list):
        return [without_keys(item, excluded_names, excluded_prefixes) for item in value]
    return deepcopy(value)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def implementation_variant(implementation: Dict[str, Any]) -> str:
    library = implementation.get("library")
    device = implementation.get("device")
    if device in (None, "default"):
        return library
    if library == "sklearn":
        data_library = implementation.get("data_library")
        if data_library is not None:
            return f"{library}-{data_library}-{device}"
    return f"{library}-{device}"


def case_name(case: Dict[str, Any]) -> str:
    algorithm = case.get("algorithm", {})
    return algorithm.get("estimator") or algorithm.get("function") or "unknown"
