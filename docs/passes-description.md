## triton-rewrite-tensor-pointer
- This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute the pointer/mask/other for each load/store.

- make_tensor_ptr and advance are helpers operations to make pointer operations easier

## reorder-broadcast
- elementwise(broadcast(a)) => broadcast(elementwise(a))
- elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))

## barriers
barriers are added after convert-triton-gpu-to-LLVM

## Triton Instrument
ConSan instruments Triton IR to detect illegal concurrent accesses to shared and Tensor Core memory under warp specialization. 

## Blocked Layout
- GPU IR uses a blocked encoding
- An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout used to promote memory coalescing in LoadInst and StoreInst.

Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}>