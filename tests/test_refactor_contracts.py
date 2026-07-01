import json
from datetime import datetime, timezone

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from reporting.matching import (
    BenchmarkRecord,
    Match,
    append_iterations_warning,
    method_results_from_records,
    read_benchmark_records,
)
from sklbench.config import load_cases_from_script, validate_case
from sklbench.orchestrator import implementation
from sklbench.runner.__main__ import _as_jsonable, estimator_params_for_repeat


def minimal_case(**overrides):
    case = {
        "bench": {
            "n_runs": 1,
            "time_limit": 600,
            "flush_cache": False,
            "memory_profiling_interval": 0.001,
        },
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


def test_validate_case_elides_default_bench_values():
    case = validate_case(
        {
            "bench": {"n_runs": 10, "time_limit": 600, "flush_cache": False},
            "implementation": {"library": "sklearn"},
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
        }
    )

    assert "bench" not in case.json_dict()


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
        "configs/hgb_scaling.py",
    ],
)
def test_active_configs_generate_valid_cases(path):
    cases = load_cases_from_script(path)

    assert cases
    assert all(case.algorithm.estimator != "DBSCAN" for case in cases)


def test_orchestrator_stores_runner_jsonl_rows(monkeypatch):
    rows = [
        {
            "case": {"bench": {"n_runs": 2}},
            "data_desc": {"fit": {"samples": 4}, "predict": {"samples": 2}},
            "time_ms": {"fit": 1.0, "predict": 0.5},
            "metrics": {"fit": {"accuracy": 1.0}, "predict": {"accuracy": 0.5}},
            "profiling_metrics": {"fit": {"cpu load[%]": [10]}},
            "attributes": {"n_iter": 2},
            "logs": {"stdout": "", "stderr": ""},
        },
        {
            "case": {"bench": {"n_runs": 2}},
            "data_desc": {"fit": {"samples": 4}, "predict": {"samples": 2}},
            "time_ms": {"fit": 1.5, "predict": 0.75},
            "metrics": {"fit": {"accuracy": 1.0}, "predict": {"accuracy": 0.5}},
            "profiling_metrics": {"fit": {"cpu load[%]": [20]}},
            "attributes": {"n_iter": 2},
            "logs": {"stdout": "", "stderr": ""},
        },
    ]

    def run_runner_from_case(bench_case, log_level):
        return 0, rows, None

    monkeypatch.setattr(implementation, "run_runner_from_case", run_runner_from_case)
    _, results, failed_cases = implementation.call_benchmarks(
        [validate_case(minimal_case())]
    )

    assert failed_cases == []
    assert results == [
        {"case": validate_case(minimal_case()).json_dict(), "results": rows}
    ]


def test_runner_serializes_small_array_like_attributes():
    class ArrayLike:
        shape = (1,)
        dtype = "torch.int32"

        def tolist(self):
            return [19]

    assert _as_jsonable(ArrayLike()) == 19


def test_runner_serializes_large_array_like_attributes_as_metadata():
    class ArrayLike:
        shape = (17,)
        dtype = "torch.float64"

        def tolist(self):
            raise AssertionError("large arrays should not be materialized")

    assert _as_jsonable(ArrayLike()) == {"shape": [17], "dtype": "torch.float64"}


def make_record(runs):
    case = validate_case(minimal_case()).json_dict()
    case.pop("bench", None)
    return BenchmarkRecord(
        hardware_hash="hardware",
        software_hash="software",
        timestamp_recorded=datetime(2026, 1, 1, tzinfo=timezone.utc),
        case=case,
        runs=runs,
    )


def make_run(
    *,
    fit_time=1.0,
    predict_time=0.5,
    fit_metric=1.0,
    predict_metric=0.5,
    data_desc=None,
    attributes=None,
    logs=None,
    profiling_metrics=None,
):
    return {
        "data_desc": data_desc
        or {"fit": {"samples": 4}, "predict": {"samples": 2}},
        "time_ms": {"fit": fit_time, "predict": predict_time},
        "metrics": {
            "fit": {"R2": fit_metric},
            "predict": {"R2": predict_metric},
        },
        "profiling_metrics": profiling_metrics or {"fit": {"cpu load[%]": [10]}},
        "attributes": attributes or {},
        "logs": logs or {"stdout": "", "stderr": ""},
    }


def test_read_benchmark_records_reads_raw_results(tmp_path):
    result_path = tmp_path / "20260101T010203000004Z.json"
    case = validate_case(minimal_case()).json_dict()
    result_path.write_text(
        json.dumps(
            {
                "hardware_hash": "hardware",
                "software_hash": "software",
                "bench_cases": [{"case": case, "results": [make_run()]}],
                "failed_cases": [],
            }
        ),
        encoding="utf-8",
    )

    records = read_benchmark_records(tmp_path)
    expected_case = validate_case(minimal_case()).json_dict()
    expected_case.pop("bench", None)

    assert len(records) == 1
    assert records[0].case == expected_case
    assert records[0].runs == [make_run()]


def test_reporting_reads_raw_runner_results():
    results = method_results_from_records(
        [
            make_record(
                [
                    make_run(
                        fit_time=1.0,
                        predict_time=0.5,
                        attributes={"n_iter": 2},
                    ),
                    make_run(
                        fit_time=1.5,
                        predict_time=0.75,
                        attributes={"n_iter": 2},
                    ),
                ]
            )
        ]
    )

    fit_result = next(result for result in results if result.method == "fit")
    predict_result = next(result for result in results if result.method == "predict")

    assert fit_result.times == [1.0, 1.5]
    assert fit_result.metric_samples["fit"]["R2"] == [1.0, 1.0]
    assert fit_result.metrics["fit"]["R2"] == 1.0
    assert "cpu load[%]" not in fit_result.metrics["fit"]
    assert fit_result.attributes == {"iterations": [2]}
    assert predict_result.times == [0.5, 0.75]
    assert predict_result.data_desc == {"samples": 2}


def test_reporting_rejects_data_desc_drift_across_repeats():
    record = make_record(
        [
            make_run(data_desc={"fit": {"samples": 4}, "predict": {"samples": 2}}),
            make_run(data_desc={"fit": {"samples": 5}, "predict": {"samples": 2}}),
        ]
    )

    with pytest.raises(ValueError, match="Inconsistent data_desc"):
        method_results_from_records([record])


def test_reporting_preserves_first_non_empty_logs():
    results = method_results_from_records(
        [
            make_record(
                [
                    make_run(logs={"stdout": "", "stderr": ""}),
                    make_run(logs={"stdout": "", "stderr": "warning"}),
                    make_run(logs={"stdout": "later", "stderr": ""}),
                ]
            )
        ]
    )

    assert results[0].logs == {"stdout": "", "stderr": "warning"}


def test_metric_matching_uses_baseline_variability_and_candidate_mean():
    base_result = next(
        result
        for result in method_results_from_records(
            [
                make_record(
                    [
                        make_run(fit_metric=1.0),
                        make_run(fit_metric=1.0),
                        make_run(fit_metric=1.0),
                    ]
                )
            ]
        )
        if result.method == "fit"
    )
    candidate_result = next(
        result
        for result in method_results_from_records(
            [
                make_record(
                    [
                        make_run(fit_metric=1.0),
                        make_run(fit_metric=1.0),
                        make_run(fit_metric=1.2),
                    ]
                )
            ]
        )
        if result.method == "fit"
    )

    match = Match(base_result, candidate_result, warnings=[])

    assert match.metrics_differences
    assert "target_mean" in match.metrics_differences[0]


def test_metric_matching_uses_three_sigma_tolerance_floor():
    base_result = next(
        result
        for result in method_results_from_records(
            [
                make_record(
                    [
                        make_run(fit_metric=0.98),
                        make_run(fit_metric=1.0),
                        make_run(fit_metric=1.02),
                    ]
                )
            ]
        )
        if result.method == "fit"
    )
    candidate_result = next(
        result
        for result in method_results_from_records(
            [make_record([make_run(fit_metric=1.05)])]
        )
        if result.method == "fit"
    )

    assert Match(base_result, candidate_result, warnings=[]).metrics_match


def test_iteration_warning_uses_selected_attributes_only():
    base_result = next(
        result
        for result in method_results_from_records(
            [make_record([make_run(attributes={"solver": "svd", "n_iter": 2})])]
        )
        if result.method == "fit"
    )
    candidate_result = next(
        result
        for result in method_results_from_records(
            [make_record([make_run(attributes={"solver": "lbfgs", "n_iter": 3})])]
        )
        if result.method == "fit"
    )
    warnings = []

    append_iterations_warning(base_result, candidate_result, warnings)

    assert len(warnings) == 1
    assert base_result.attributes == {"iterations": [2]}
    assert candidate_result.attributes == {"iterations": [3]}


def test_estimator_params_for_repeat_sets_supported_random_state():
    params = estimator_params_for_repeat(RandomForestClassifier, {"n_estimators": 2}, 3)

    assert params == {"n_estimators": 2, "random_state": 3}


def test_estimator_params_for_repeat_skips_unsupported_random_state():
    params = estimator_params_for_repeat(LinearRegression, {}, 3)

    assert params == {}


def test_estimator_params_for_repeat_preserves_explicit_random_state():
    params = estimator_params_for_repeat(
        RandomForestClassifier, {"random_state": 42}, 3
    )

    assert params == {"random_state": 42}
