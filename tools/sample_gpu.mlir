#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
//   func.func @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("examples/kernels/binary_ops.py":73:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("examples/kernels/binary_ops.py":73:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("examples/kernels/binary_ops.py":73:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("examples/kernels/binary_ops.py":73:0)) attributes {noinline = false} {
    
//     %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    
//     %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
//     tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
//     tt.return
//   }
// }