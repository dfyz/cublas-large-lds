# "No overlap" scenario

The most straightforward way to trigger an OOB access in one of cuBLAS kernels is to:
  * allocate a small memory chunk for matrix `a`;
  * allocate huge memory chunks for matrices `b` and `c`;
  * run a GEMM with relatively small `m`, `n`, `k` and `lda`, but huge `ldb` and `ldc`.

However, it looks like only CUDA 11.x is affected — I failed to reproduce the crash with CUDA 10.x.

Note that the matrices are pretty large, and you need a GPU with > 16 GB of memory (we used `Tesla V100-PCIE-32GB`). Here's an example output of `cuda-memcheck` (note that the kernel is trying to access `0x7fb5df82277e`, which is not in any of the chunks we allocated):
```
> make CUDA_VERSION=11.3 
/usr/local/cuda-11.3/bin/nvcc -g -std=c++14 -lcublas -O2 main.cpp -o main
> CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stderr /usr/local/cuda-11.3/bin/cuda-memcheck ./main no_overlap
Allocated a chunk: [7fbb9e200000; 7fbb9e800000)
Allocated a chunk: [7fb91e000000; 7fbb8d7c8000)
Allocated a chunk: [7fb6ae000000; 7fb91d7c8000)
[...]
I! cuBLAS (v11.0) function cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t) called:
[...]
i!  m: type=int; val=1024
i!  n: type=int; val=100
i!  k: type=int; val=3072
[...]
i!  A: type=void; val=POINTER (IN HEX:0x0x7fbb9e200000)
[...]
i!  lda: type=int; val=1024
i!  B: type=void; val=POINTER (IN HEX:0x0x7fb91e000000)
[...]
i!  ldb: type=int; val=52301824
[...]
i!  C: type=void; val=POINTER (IN HEX:0x0x7fb6ae000000)
[...]
i!  ldc: type=int; val=52301824
[...]
========= CUDA-MEMCHECK
========= Invalid __global__ write of size 2
=========     at 0x00000e60 in void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *)
=========     by thread (63,0,0) in block (399,0,0)
=========     Address 0x7fb5df82277e is out of bounds
=========     Device Frame:void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *) (void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *) : 0xe60)
[...]
```

# "Overlap" scenario

This is basically the same GEMM, but with rows of `b` and `c` interleaved in a single huge memory chunk:
```
          ▲ ┌─────────────┐▲
          │ │             ││
          │ │  Matrix b   ││ 1024 rows
          │ │             ││     k
          │ ├─────────────┤▼
          │ │             │▲
          │ │  Matrix c   ││
          │ │             ││ 3072 rows
          │ │             ││     m
≈50M rows │ │             ││           
 ld{b,c}  │ ├─────────────┤▼           
          │ │             │            
          │ │             │            
          │ │             │            
          │ │             │            
          │ │             │            
          ▼ └─────────────┘            
```

This is a somewhat unusual pattern, but we use it extensively in our codebase, and it works like a charm for smaller `ld{b,c}` values. I don't see anything in the documentation that would specifically prohibit this kind of memory layout for GEMM.

This crashes cuBLAS kernels both from CUDA 10.x and CUDA 11.x, although with slightly different combinations of `m` and `k`.

An example of output for CUDA 10.x:
```
> make CUDA_VERSION=10.1                                                                                                
/usr/local/cuda-10.1/bin/nvcc -g -std=c++14 -lcublas -O2 main.cpp -o main
> CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stderr /usr/local/cuda-10.1/bin/cuda-memcheck ./main overlap
Allocated a chunk: [7f879e200000; 7f879e800000)
Allocated a chunk: [7f851e000000; 7f878d7c8000)
I! cuBLAS (v10.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
[...]
========= CUDA-MEMCHECK
========= Invalid __global__ write of size 8
=========     at 0x000023e0 in volta_h884gemm_64x64_ldg8_nn
=========     by thread (63,0,0) in block (31,1,0)
=========     Address 0x7f89874077b8 is out of bounds
```

An example of output for CUDA 11.x:
```
> make CUDA_VERSION=11.3                                                                                             
/usr/local/cuda-11.3/bin/nvcc -g -std=c++14 -lcublas -O2 main.cpp -o main
> CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stderr /usr/local/cuda-11.3/bin/cuda-memcheck ./main overlap
Allocated a chunk: [7fb504200000; 7fb504800000)
Allocated a chunk: [7fb284000000; 7fb4f37c8000)
[...]
I! cuBLAS (v11.0) function cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t) called:
[...]
i!  m: type=int; val=1024
i!  n: type=int; val=100
i!  k: type=int; val=3072
[...]
i!  A: type=void; val=POINTER (IN HEX:0x0x7fb504200000)
[...]
i!  lda: type=int; val=1024
i!  B: type=void; val=POINTER (IN HEX:0x0x7fb284000000)
[...]
i!  ldb: type=int; val=52301824
[...]
i!  C: type=void; val=POINTER (IN HEX:0x0x7fb284001800)
[...]
i!  ldc: type=int; val=52301824
[...]
========= CUDA-MEMCHECK
========= Invalid __global__ write of size 2
=========     at 0x00000e60 in void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *)
=========     by thread (95,0,0) in block (399,0,0)
=========     Address 0x7fb1b5823fbe is out of bounds
=========     Device Frame:void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *) (void splitKreduce_kernel<__half, __half, __half, __half>(cublasSplitKParams<__half>, __half const *, __half const *, __half*, __half const *, __half const *, __half const *) : 0xe60)
```
