## triton-rewrite-tensor-pointer
- This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute the pointer/mask/other for each load/store.

- make_tensor_ptr and advance are helpers operations to make pointer operations easier

## reorder-broadcast
- elementwise(broadcast(a)) => broadcast(elementwise(a))
- elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
