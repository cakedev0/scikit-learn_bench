# Machine Learning Benchmarks

[![Build Status](https://dev.azure.com/daal/scikit-learn_bench/_apis/build/status/IntelPython.scikit-learn_bench?branchName=main)](https://dev.azure.com/daal/scikit-learn_bench/_build/latest?definitionId=8&branchName=main)

**Scikit-learn_bench** is a benchmark tool for libraries and frameworks implementing Scikit-learn-like APIs and other workloads.

Benefits:
- Full control of benchmarks suite through CLI
- Flexible and powerful benchmark config structure
- Available with advanced profiling tools, such as Intel(R) VTune* Profiler

### 📜 Table of Contents

- [Machine Learning Benchmarks](#machine-learning-benchmarks)
  - [🔧 Create a Python Environment](#-create-a-python-environment)
  - [🚀 How To Use Scikit-learn\_bench](#-how-to-use-scikit-learn_bench)
    - [Benchmarks Runner](#benchmarks-runner)
    - [Scikit-learn\_bench High-Level Workflow](#scikit-learn_bench-high-level-workflow)
  - [📚 Benchmark Types](#-benchmark-types)
  - [📑 Documentation](#-documentation)

## 🔧 Create a Python Environment

How to create a usable Python environment with the following required frameworks:

- **sklearn, sklearnex, and gradient boosting frameworks**:

```bash
# with pip
pip install -r envs/requirements-sklearn.txt
# or with conda
conda env create -n sklearn -f envs/conda-env-sklearn.yml
```

- **RAPIDS**:

```bash
conda env create -n rapids --solver=libmamba -f envs/conda-env-rapids.yml
```

## 🚀 How To Use Scikit-learn_bench

### Benchmarks Runner

How to run benchmarks using the `sklbench` module and a specific configuration:

```bash
python -m sklbench --config configs/sklearn_example.json
```

The default output is an append-only `results/` directory containing flat,
timestamped benchmark result files with embedded environment metadata. To specify
a custom output directory, run:

```bash
python -m sklbench --config configs/sklearn_example.json --results-dir results_example
```

For a description of all benchmarks runner arguments, refer to [documentation](sklbench/runner/README.md#arguments).

### Scikit-learn_bench High-Level Workflow

```mermaid
flowchart TB
    A[User] -- High-level arguments --> B[Benchmarks runner]
    B -- Generated benchmarking cases --> C["Benchmarks collection"]
    C -- Raw JSON-formatted results --> A

    classDef userStyle fill:#44b,color:white,stroke-width:2px,stroke:white;
    class A userStyle
```

## 📚 Benchmark Types

**Scikit-learn_bench** supports the following types of benchmarks:

 - **Scikit-learn estimator** - Measures performance and quality metrics of the [sklearn-like estimator](https://scikit-learn.org/stable/glossary.html#term-estimator).
 - **Function** - Measures performance metrics of specified function.

## 📑 Documentation
[Scikit-learn_bench](README.md):
- [Configs](configs/README.md)
  - [Benchmarking Config Specification](configs/BENCH-CONFIG-SPEC.md)
- [Benchmarks Runner](sklbench/runner/README.md)
- [Reports](sklbench/reports/README.md)
- [Benchmarks](sklbench/benchmarks/README.md)
- [Data Processing and Storage](sklbench/datasets/README.md)
- [Emulators](sklbench/emulators/README.md)
- [Developer Guide](docs/README.md)
