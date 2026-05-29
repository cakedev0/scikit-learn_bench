# Reports

`sklbench.reports` builds static reports from append-only result files.

## Speed-Ups

Generate an interactive HTML speed-up report and a pandas-friendly CSV sidecar:

```bash
python -m sklbench.reports speedups \
  --output speedups.html \
  results/69cd73/results_20260529T120640Z.json
```

The CSV defaults to the HTML output path with a `.csv` suffix. Use
`--csv-output path/to/speedups.csv` to choose a different path.

Inputs are baseline `results_<datetime>.json` files. Each input file must be
located in a result directory with a matching environment file at
`../envs/<env_name>.json`.

If the baseline inputs contain exactly one implementation variant, it is used as
the base automatically. If they contain multiple variants, pass `--base`.
Comparison variants are discovered by scanning sibling result files under the
same `results/` root and keeping files with matching `environment.hardware`.

Implementation variants are named as `<library>` or `<library>-<device>`.
For non-default `sklearn` array API variants, the data format is included as
`<library>-<format>-<device>`. `device: null` and `device: "default"` are both
rendered as `<library>`.
