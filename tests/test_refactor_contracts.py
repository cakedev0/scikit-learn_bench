import json

import pytest

from sklbench.config import load_cases_from_script, validate_case
from sklbench.orchestrator.implementation import aggregate_runner_rows


def minimal_case(**overrides):
    case = {
        "bench": {"n_runs": 1, "time_limit": None},
        "implementation": {"library": "sklearn", "device": None},
        "algorithm": {
            "estimator": "Ridge",
            "estimator_params": {},
        },
        "data": {
            "source": "make_regression",
            "generation_kwargs": {"n_samples": 10, "n_features": 3},
            "split_kwargs": {"test_size": 0.2},
        },
    }
    case.update(overrides)
    return case


def test_validate_case_returns_model_with_normalized_json_dict():
    case = validate_case(minimal_case())
    case_dict = case.json_dict()

    assert case.implementation.library == "sklearn"
    assert case_dict["implementation"] == {"library": "sklearn"}
    assert case_dict["bench"] == {"n_runs": 1}
    assert json.loads(json.dumps(case_dict, allow_nan=False)) == case_dict
    assert case.data.name(shortened=True) == "make_regr"


def test_validate_case_rejects_invalid_top_level_sections():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_case(minimal_case(extra={}))


def test_validate_case_rejects_invalid_nested_section_keys():
    case = minimal_case()
    case["data"]["typo"] = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_case(case)


def test_validate_case_rejects_non_json_values():
    case = minimal_case()
    case["algorithm"]["estimator_params"]["callback"] = lambda value: value

    with pytest.raises(ValueError, match="JSON serializable"):
        validate_case(case)


def test_load_cases_from_script_discovers_generate_cases(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        """
def generate_cases():
    return [{
        "bench": {"n_runs": 1},
        "implementation": {"library": "sklearn"},
        "algorithm": {"estimator": "Ridge"},
        "data": {"source": "make_regression"},
    }]
""",
        encoding="utf-8",
    )

    cases = load_cases_from_script(config)

    assert len(cases) == 1
    assert cases[0].algorithm.estimator == "Ridge"


def test_load_cases_from_script_rejects_invalid_return_type(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("def generate_cases():\n    return {}\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must return a list"):
        load_cases_from_script(config)


def test_load_cases_from_script_reports_case_index(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        """
def generate_cases():
    return [
        {
            "implementation": {"library": "sklearn"},
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
        },
        {"bad": "case"},
    ]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid case at index 1"):
        load_cases_from_script(config)


@pytest.mark.parametrize(
    "path",
    [
        "configs/sklearn.py",
        "configs/sklearnex_cpu.py",
        "configs/sklearnex_gpu.py",
        "configs/array_api_cpu.py",
        "configs/array_api_intel.py",
        "configs/array_api_nvidia.py",
    ],
)
def test_active_configs_generate_valid_cases(path):
    cases = load_cases_from_script(path)

    assert cases
    assert all(case.algorithm.estimator != "DBSCAN" for case in cases)


def test_orchestrator_aggregates_runner_jsonl_rows():
    rows = [
        {
            "case": {"bench": {"n_runs": 2}},
            "repeat": 0,
            "data_desc": {"fit": {"samples": 4}, "predict": {"samples": 2}},
            "time_ms": {"fit": 1.0, "predict": 0.5},
            "metrics": {"fit": {"accuracy": 1.0}, "predict": {"accuracy": 0.5}},
            "execution_metrics": {"fit": {"cpu load[%]": [10]}},
            "attributes": {"n_iter": 2},
            "logs": {"stdout": "", "stderr": ""},
        },
        {
            "case": {"bench": {"n_runs": 2}},
            "repeat": 1,
            "data_desc": {"fit": {"samples": 4}, "predict": {"samples": 2}},
            "time_ms": {"fit": 1.5, "predict": 0.75},
            "metrics": {"fit": {"accuracy": 1.0}, "predict": {"accuracy": 0.5}},
            "execution_metrics": {"fit": {"cpu load[%]": [20]}},
            "attributes": {"n_iter": 2},
            "logs": {"stdout": "", "stderr": ""},
        },
    ]

    result = aggregate_runner_rows(rows)

    assert result["time[ms]"] == {"fit": [1.0, 1.5], "predict": [0.5, 0.75]}
    assert result["metrics"]["fit"]["cpu load[%]"] == [10, 20]
    assert result["metrics"]["fit"]["n_iter"] == 2
    assert result["data_desc"]["predict"]["samples"] == 2
