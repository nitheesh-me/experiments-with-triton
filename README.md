# experiments-with-triton

This repository contains small experiments with Triton kernels and a helper
CLI `triton2ir.py` that can run a Python module which compiles Triton kernels
and capture or replay the Triton IR output.

Usage
-----

Run a module and capture its printed IR (captures stdout+stderr):

```bash
python3 triton2ir.py sample.py --save-ir sample_ir.txt
```

Read a previously saved IR file without running the module (no Triton/CUDA
needed). Note: the CLI expects a MODULE positional argument; pass any existing
module path (it will be ignored when --read-ir is used):

```bash
python3 triton2ir.py sample.py --read-ir sample_ir.txt
```

If the callable you ask the CLI to invoke returns a string (or bytes), the
CLI will prefer writing that returned value exactly to the save file. This is
handy when your module exposes a helper that returns the `compiled_kernel.asm['ttir']`.

Examples
--------

- `sample.py` — example that compiles a simple add kernel and prints the IR at import time.
- `examples/return_ir.py` — small test module whose function `return_ir()` returns a string; `triton2ir.py --save-ir` will save the returned string exactly.

Notes
-----

- `--save-ir` saves the captured stdout+stderr unless the invoked callable returns a string/bytes; in that case that returned value is saved instead.
- `--read-ir` avoids running the module and just prints the saved IR file.
- For the CLI UX install `click` (optional): `pip install click`.
- Triton/PyTorch/CUDA are required only if you run a module that actually compiles kernels.

Next steps
----------

If you want, I can:

- Add a `--cache PATH` mode that runs the module only if the cache file is missing.
- Allow passing arguments to the callable (e.g., via JSON on the command line).
- Add structured metadata (JSON) for multiple IRs per module.

