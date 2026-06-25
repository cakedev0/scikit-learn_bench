# Benchmark Runner

`sklbench.runner` executes one already-expanded benchmark case.

It is intentionally separate from config parsing and orchestration:

- a Python config script generates validated benchmark cases;
- the orchestrator records environments, launches runner subprocesses, captures
  logs/errors, and writes result files;
- the runner validates one case file, loads data, runs repetitions, and writes
  JSONL.

## CLI Contract

```bash
python -m sklbench.runner \
  --case-file /path/to/case.json \
  --output-jsonl /tmp/sklbench-result.jsonl \
  --log-level WARNING
```

The output file contains one JSON object per repetition. Each line includes the
case, repetition index, data description, per-method timings in milliseconds,
metrics, model attributes, and warnings.
