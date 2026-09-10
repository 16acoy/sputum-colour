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


## Modules