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

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

from .common import (
    RESULT_FILE_RE,
    ResultFile,
    case_name,
    implementation_variant,
    load_result_file,
    load_result_files,
    stable_json,
    without_keys,
)


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


@dataclass
class Record:
    variant: str
    method: str
    name: str
    match_key: str
    median_time: float
    result_file: ResultFile
    case: Dict[str, Any]


@dataclass
class Point:
    target_variant: str
    method: str
    name: str
    case: Dict[str, Any]
    base_result_file: Path
    target_result_file: Path
    log2_speedup: float
    speedup: float
    base_time: float
    target_time: float
    base_timestamp: datetime
    target_timestamp: datetime
    stale: bool = False


def add_speedups_parser(parser):
    parser.add_argument(
        "--base",
        help="Base implementation variant, e.g. sklearn or sklearnex-gpu.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="speedups.html",
        type=Path,
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help=(
            "Output CSV file path. Defaults to the HTML output path with a "
            ".csv suffix."
        ),
    )
    parser.add_argument(
        "--stale-days",
        default=7,
        type=float,
        help=(
            "Mark variant pairs stale when selected matches span more than this "
            "many days."
        ),
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="Input result files ending with _<datetime>.json.",
    )


def match_key(case: Dict[str, Any], environment: Dict[str, Any]) -> str:
    algorithm = without_keys(
        case.get("algorithm", {}),
        excluded_names={"library", "device", "sklearnex_context"},
        excluded_prefixes=("sklearn_context",),
    )
    estimator_params = algorithm.get("estimator_params")
    if isinstance(estimator_params, dict):
        estimator_params.pop("max_bins", None)
    data = without_keys(case.get("data", {}), excluded_names={"format"})
    return stable_json(
        {
            "hardware": environment.get("hardware", {}),
            "algorithm": algorithm,
            "data": data,
        }
    )


def flatten_records(result_files: List[ResultFile]) -> List[Record]:
    records = []
    for result_file in result_files:
        for bench_case in result_file.bench_cases:
            case = bench_case.get("case", {})
            algorithm = case.get("algorithm", {})
            variant = implementation_variant(algorithm, case.get("data", {}))
            name = case_name(case)
            key = match_key(case, result_file.environment)
            for method, result in bench_case.get("results", {}).items():
                times = result.get("time[ms]")
                if not isinstance(times, list) or len(times) == 0:
                    continue
                records.append(
                    Record(
                        variant=variant,
                        method=method,
                        name=name,
                        match_key=key,
                        median_time=float(median(times)),
                        result_file=result_file,
                        case=case,
                    )
                )
    return records


def ignored_match_params_key(case: Dict[str, Any]) -> str:
    estimator_params = case.get("algorithm", {}).get("estimator_params", {})
    if not isinstance(estimator_params, dict):
        estimator_params = {}
    return stable_json({"max_bins": estimator_params.get("max_bins")})


def keep_latest_records(records: List[Record]) -> Dict[Tuple[str, str, str, str], Record]:
    latest = {}
    for record in records:
        key = (
            record.match_key,
            record.variant,
            record.method,
            ignored_match_params_key(record.case),
        )
        current = latest.get(key)
        if (
            current is None
            or record.result_file.timestamp > current.result_file.timestamp
        ):
            latest[key] = record
    return latest


def keep_latest_base_records(records: List[Record]) -> Dict[Tuple[str, str], Record]:
    latest = {}
    for record in records:
        key = (record.match_key, record.method)
        current = latest.get(key)
        if (
            current is None
            or record.result_file.timestamp > current.result_file.timestamp
        ):
            latest[key] = record
    return latest


def hardware_key(result_file: ResultFile) -> str:
    return stable_json(result_file.environment.get("hardware", {}))


def discover_result_files(base_result_files: List[ResultFile]) -> List[ResultFile]:
    base_hardware_keys = {hardware_key(result_file) for result_file in base_result_files}
    if len(base_hardware_keys) != 1:
        raise ValueError("Input base result files must all use the same hardware")
    base_hardware_key = next(iter(base_hardware_keys))

    result_files = []
    seen_paths = set()
    roots = sorted({result_file.path.parent.parent for result_file in base_result_files})
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            if (
                {"envs", "hardware-envs", "software-envs"} & set(path.parts)
                or path in seen_paths
            ):
                continue
            if not RESULT_FILE_RE.match(path.name):
                continue
            result_file = load_result_file(path)
            if hardware_key(result_file) != base_hardware_key:
                continue
            result_files.append(result_file)
            seen_paths.add(path)
    return result_files


def select_base_variant(records: List[Record], requested_base: str | None) -> str:
    variants = sorted({record.variant for record in records})
    if requested_base is not None:
        if requested_base not in variants:
            raise ValueError(
                f"Base variant '{requested_base}' is not present in input files. "
                f"Available variants: {', '.join(variants)}"
            )
        return requested_base
    if len(variants) == 1:
        return variants[0]
    raise ValueError(
        "Input files contain multiple implementation variants; pass --base. "
        f"Available variants: {', '.join(variants)}"
    )


def build_points(
    latest_base_records: Dict[Tuple[str, str], Record],
    latest_target_records: Dict[Tuple[str, str, str, str], Record],
    base_variant: str,
    stale_days: float,
) -> Tuple[List[Point], int, List[Dict[str, Any]]]:
    points = []
    unmatched = 0
    pair_timestamps = {}

    for (key, variant, method, _ignored_params), target_record in (
        latest_target_records.items()
    ):
        if variant == base_variant:
            continue
        base_record = latest_base_records.get((key, method))
        if base_record is None:
            unmatched += 1
            continue
        if target_record.median_time == 0:
            unmatched += 1
            continue
        speedup = base_record.median_time / target_record.median_time
        point = Point(
            target_variant=variant,
            method=method,
            name=target_record.name,
            case=target_record.case,
            base_result_file=base_record.result_file.path,
            target_result_file=target_record.result_file.path,
            log2_speedup=math.log2(speedup),
            speedup=speedup,
            base_time=base_record.median_time,
            target_time=target_record.median_time,
            base_timestamp=base_record.result_file.timestamp,
            target_timestamp=target_record.result_file.timestamp,
        )
        points.append(point)
        pair_key = (base_variant, variant)
        pair_timestamps.setdefault(pair_key, []).extend(
            [base_record.result_file.timestamp, target_record.result_file.timestamp]
        )

    stale_pairs = []
    stale_delta = stale_days * 24 * 60 * 60
    for pair_key, timestamps in pair_timestamps.items():
        oldest = min(timestamps)
        newest = max(timestamps)
        if (newest - oldest).total_seconds() > stale_delta:
            stale_pairs.append(
                {
                    "base": pair_key[0],
                    "target": pair_key[1],
                    "oldest": oldest.isoformat(),
                    "newest": newest.isoformat(),
                }
            )

    stale_pair_keys = {(pair["base"], pair["target"]) for pair in stale_pairs}
    for point in points:
        point.stale = (base_variant, point.target_variant) in stale_pair_keys

    return points, unmatched, stale_pairs


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hover_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def hover_lines(value: Any, prefix: str = "") -> List[str]:
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines = []
        for key, nested_value in sorted(value.items()):
            if isinstance(nested_value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(hover_lines(nested_value, prefix=f"{prefix}  "))
            elif isinstance(nested_value, list):
                lines.append(
                    f"{prefix}{key}: {json.dumps(nested_value, default=str)}"
                )
            else:
                lines.append(f"{prefix}{key}: {hover_value(nested_value)}")
        return lines
    return [f"{prefix}{hover_value(value)}"]


def hover_text(value: Any) -> str:
    return "<br>".join(hover_lines(value))


def point_hover(point: Point) -> str:
    algorithm = point.case.get("algorithm", {})
    data = data_params_for_display(point.case)
    return (
        "<b>estimator params</b><br>"
        f"{hover_text(algorithm.get('estimator_params', {}))}<br>"
        "<b>data params</b><br>"
        f"{hover_text(data)}"
    )


def data_params_for_display(case: Dict[str, Any]) -> Dict[str, Any]:
    return without_keys(
        case.get("data", {}),
        excluded_names={"training", "inference"},
    )


def csv_output_path(html_output: Path, requested_csv_output: Path | None) -> Path:
    if requested_csv_output is not None:
        return requested_csv_output
    return html_output.with_suffix(".csv")


def csv_rows(points: List[Point], base_variant: str) -> List[Dict[str, Any]]:
    rows = []
    for point in sorted(
        points,
        key=lambda point: (point.method, point.name, point.target_variant),
    ):
        algorithm = point.case.get("algorithm", {})
        data = data_params_for_display(point.case)
        rows.append(
            {
                "base_variant": base_variant,
                "target_variant": point.target_variant,
                "method": point.method,
                "estimator": point.name,
                "speedup": point.speedup,
                "log2_speedup": point.log2_speedup,
                "base_median_time_ms": point.base_time,
                "target_median_time_ms": point.target_time,
                "stale": point.stale,
                "base_timestamp": point.base_timestamp.isoformat(),
                "target_timestamp": point.target_timestamp.isoformat(),
                "base_result_file": str(point.base_result_file),
                "target_result_file": str(point.target_result_file),
                "algorithm_task": algorithm.get("task"),
                "target_library": algorithm.get("library"),
                "target_device": algorithm.get("device"),
                "data_source": data.get("source"),
                "data_format": data.get("format"),
                "estimator_params": compact_json(
                    algorithm.get("estimator_params", {})
                ),
                "data_params": compact_json(data),
                "generation_kwargs": compact_json(
                    data.get("generation_kwargs", {})
                ),
                "split_kwargs": compact_json(data.get("split_kwargs", {})),
            }
        )
    return rows


def write_csv_report(points: List[Point], base_variant: str, path: Path) -> None:
    rows = csv_rows(points, base_variant)
    fieldnames = [
        "base_variant",
        "target_variant",
        "method",
        "estimator",
        "speedup",
        "log2_speedup",
        "base_median_time_ms",
        "target_median_time_ms",
        "stale",
        "base_timestamp",
        "target_timestamp",
        "base_result_file",
        "target_result_file",
        "algorithm_task",
        "target_library",
        "target_device",
        "data_source",
        "data_format",
        "estimator_params",
        "data_params",
        "generation_kwargs",
        "split_kwargs",
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def variant_offsets(variants: List[str]) -> Dict[str, float]:
    if len(variants) <= 1:
        return {variant: 0 for variant in variants}
    step = 0.14
    center = (len(variants) - 1) / 2
    return {
        variant: (index - center) * step for index, variant in enumerate(variants)
    }


def point_has_max_bins(point: Point) -> bool:
    estimator_params = point.case.get("algorithm", {}).get("estimator_params", {})
    return isinstance(estimator_params, dict) and "max_bins" in estimator_params


def mixed_max_bins_columns(points: List[Point], method: str) -> set[Tuple[str, str]]:
    column_has_max_bins = {}
    for point in points:
        if point.method != method:
            continue
        key = (point.name, point.target_variant)
        column_has_max_bins.setdefault(key, set()).add(point_has_max_bins(point))
    return {
        key
        for key, has_max_bins_values in column_has_max_bins.items()
        if has_max_bins_values == {False, True}
    }


def build_traces(
    points: List[Point],
    method: str,
    estimator_positions: Dict[str, int],
    offsets: Dict[str, float],
) -> List[Dict[str, Any]]:
    mixed_columns = mixed_max_bins_columns(points, method)
    grouped = {}
    for point in points:
        if point.method != method:
            continue
        grouped.setdefault((point.target_variant, point.stale), []).append(point)

    traces = []
    for (variant, stale), group_points in sorted(grouped.items()):
        group_points = sorted(
            group_points, key=lambda point: (point.name, point.target_variant)
        )
        marker_symbols = [
            "x"
            if point_has_max_bins(point)
            and (point.name, point.target_variant) in mixed_columns
            else "circle"
            for point in group_points
        ]
        marker = {"size": 10, "symbol": marker_symbols}
        if stale:
            marker["color"] = "rgba(220, 20, 20, 0.45)"
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": variant + (" [stale]" if stale else ""),
                "x": [
                    estimator_positions[point.name] + offsets[point.target_variant]
                    for point in group_points
                ],
                "y": [point.log2_speedup for point in group_points],
                "text": [point_hover(point) for point in group_points],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": marker,
            }
        )
    return traces


def build_layout(
    base_variant: str,
    method: str,
    estimators: List[str],
    points: List[Point],
) -> Dict[str, Any]:
    values = [point.log2_speedup for point in points if point.method == method]
    if values:
        min_tick = math.floor(min(min(values), 0))
        max_tick = math.ceil(max(max(values), 0))
    else:
        min_tick = 0
        max_tick = 0
    tick_values = list(range(min_tick, max_tick + 1))
    tick_text = [format_speedup_tick(2 ** tick) for tick in tick_values]
    return {
        "title": f"{method} speed-ups vs {base_variant}",
        "xaxis": {
            "title": "Estimator",
            "tickmode": "array",
            "tickvals": list(range(len(estimators))),
            "ticktext": estimators,
            "range": [-0.5, len(estimators) - 0.5],
        },
        "yaxis": {
            "title": "speed-up",
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": tick_text,
        },
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": 0,
                "y1": 0,
                "line": {"color": "#666", "width": 1, "dash": "dash"},
            }
        ],
        "margin": {"l": 70, "r": 30, "t": 70, "b": 120},
    }


def format_speedup_tick(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    if value >= 0.001:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def render_html(
    points: List[Point],
    stale_pairs: List[Dict[str, Any]],
    input_files: List[Path],
    discovered_files: List[Path],
    base_variant: str,
    unmatched_count: int,
) -> str:
    estimators = sorted({point.name for point in points})
    estimator_positions = {
        estimator: index for index, estimator in enumerate(estimators)
    }
    variants = sorted({point.target_variant for point in points})
    offsets = variant_offsets(variants)
    methods = [
        method
        for method in ("fit", "predict")
        if any(point.method == method for point in points)
    ]
    method_sections = []
    chart_scripts = []
    for method in methods:
        chart_id = f"{method}-chart"
        method_sections.append(
            f"<section class=\"chart-section\"><h2>{escape(method)}</h2>"
            f"<div id=\"{chart_id}\" class=\"chart\"></div></section>"
        )
        traces_json = json.dumps(
            build_traces(points, method, estimator_positions, offsets)
        ).replace("</", "<\\/")
        layout_json = json.dumps(
            build_layout(base_variant, method, estimators, points)
        ).replace("</", "<\\/")
        chart_scripts.append(
            f"Plotly.newPlot(\"{chart_id}\", {traces_json}, {layout_json}, "
            "{responsive: true});"
        )
    warnings = "\n".join(
        (
            "<li>"
            f"{escape(pair['base'])} vs {escape(pair['target'])}: "
            f"{pair['oldest']} to {pair['newest']}"
            "</li>"
        )
        for pair in stale_pairs
    )
    warning_section = (
        "<section class=\"warning\"><h2>Stale Variant Pairs</h2>"
        f"<ul>{warnings}</ul></section>"
        if stale_pairs
        else ""
    )
    input_items = "\n".join(f"<li>{escape(str(path))}</li>" for path in input_files)
    discovered_items = "\n".join(
        f"<li>{escape(str(path))}</li>" for path in discovered_files
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    safe_base_variant = escape(base_variant)
    charts_html = "\n  ".join(method_sections)
    charts_js = "\n    ".join(chart_scripts)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>sklbench speed-up report</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    .chart {{ width: 100%; height: 680px; }}
    .chart-section {{ margin-top: 28px; }}
    .summary, .warning {{ margin-bottom: 20px; }}
    .warning {{ color: #8a1f11; background: #fff4f2; padding: 12px 16px; }}
    code {{ background: #f3f3f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>sklbench speed-up report</h1>
  <section class="summary">
    <p><strong>Base variant:</strong> <code>{safe_base_variant}</code></p>
    <p><strong>Generated:</strong> {generated_at}</p>
    <p><strong>Matched points:</strong> {len(points)}</p>
    <p><strong>Unmatched target records:</strong> {unmatched_count}</p>
    <details>
      <summary>Base input files</summary>
      <ul>{input_items}</ul>
    </details>
    <details>
      <summary>Discovered comparison files</summary>
      <ul>{discovered_items}</ul>
    </details>
  </section>
  {warning_section}
  {charts_html}
  <script>
    {charts_js}
  </script>
</body>
</html>
"""


def build_speedups_report(args) -> int:
    base_result_files = load_result_files(args.result_files)
    discovered_result_files = discover_result_files(base_result_files)

    input_records = flatten_records(base_result_files)
    base_variant = select_base_variant(input_records, args.base)
    base_records = [
        record for record in input_records if record.variant == base_variant
    ]
    target_records = flatten_records(discovered_result_files)
    latest_base_records = keep_latest_base_records(base_records)
    latest_target_records = keep_latest_records(target_records)
    points, unmatched_count, stale_pairs = build_points(
        latest_base_records,
        latest_target_records,
        base_variant,
        args.stale_days,
    )
    html = render_html(
        points=points,
        stale_pairs=stale_pairs,
        input_files=args.result_files,
        discovered_files=[result_file.path for result_file in discovered_result_files],
        base_variant=base_variant,
        unmatched_count=unmatched_count,
    )
    args.output.write_text(html)
    csv_path = csv_output_path(args.output, args.csv_output)
    write_csv_report(points, base_variant, csv_path)
    print(f"Speed-up report written to {args.output}")
    print(f"Speed-up CSV written to {csv_path}")
    print(f"Matched points: {len(points)}")
    print(f"Unmatched target records: {unmatched_count}")
    return 0
