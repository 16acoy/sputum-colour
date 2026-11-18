# `sputum_colour`

This directory contains the reusable Python code used by the analysis notebooks in
this repository.

The package does not currently expose a public API from `__init__.py`; import the
modules directly:

```python
from sputum_colour import tbc
```

## Package-Level Limitations

- The package is currently an analysis support library rather than a fully
	configurable distribution package.
- The notebooks remain the primary consumers and examples of the package APIs.


## Performance diagnostics

Pipeline stages log operation timings and memory diagnostics through the
`performance.log` file in the `results_dir` passed to each stage's `main`
function. Per-image stage logs include nested operation timings; Python-traced
allocation peaks are reported for outer scopes, while nested scopes report
current allocation deltas. Process memory is reported as the process RSS
high-water mark and its increase during each operation.

## Modules
