# Tests

These tests are fully vibed.


OpenDDE numerical regressions require the launcher-applied dependency patches.
After installing or reinstalling dependencies, apply these before importing
JOpenDDE in the test process:

```bash
.venv/bin/python patches/patch_jopendde_aggregation.py
.venv/bin/python patches/patch_jopendde_padding.py
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_opendde_numerics.py tests/test_opendde_masking.py tests/test_opendde_padding.py
```

These tests use synthetic arrays and small kernels, without a checkpoint.
GPU execution of the same tests is useful for checking the reduction kernel;
passing does not establish full-model determinism or gradient correctness.
