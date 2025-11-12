## run a specific pass pipeline
```
build/cmake.linux-x86_64-cpython-3.13/bin/triton-opt sample.mlir -p='builtin.module(convert-triton-to-tritongpu{target=cuda:89})' --dump-pass-pipeline
```

## Analysis tool build command
replace: /usr/local with $(brew --prefix llvm) for mac(brew install llvm)
```
cmake -G Ninja .. \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/local/bin/clang \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir \
  -DLLVM_DIR=/usr/local/lib/cmake/llvm
```