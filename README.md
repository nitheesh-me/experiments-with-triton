# experiments-with-triton

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
