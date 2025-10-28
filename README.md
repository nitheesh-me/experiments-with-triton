# experiments-with-triton
## Environment
```
export MLIR_ENABLE_DUMP=1
export TRITON_ALWAYS_COMPILE=1
```
## Commands (FAQ)

- Capture IR:
```bash
python3 triton2ir.py sample.py --save-ir sample_ir.txt
```

- Read saved IR (no Triton/CUDA required):
```bash
python3 triton2ir.py sample.py --read-ir sample_ir.txt
```

- Extract tokens from out_return.txt to passes.txt:
```bash
grep -oP ' \(\K[^\)# ]+(?=\) )' out_return.txt > passes.txt
```
## Config Knobs
- `MLIR_ENABLE_DUMP=1` (dumps to stderr)
- `TRITON_ALWAYS_COMPILE=1` ignores cache, can't get MLIR_DUMP on repeated runs without it
- `MLIR_ENABLE_DUMP=kernelName`
<!-- - `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn. (Not sure how to use this)-->