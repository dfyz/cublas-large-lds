#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr size_t LargeStride = 52301824;

void CublasOrDie(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS failed: " + std::to_string(static_cast<int>(status)));
    }
}

void CudaOrDie(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA failed: " + std::string{cudaGetErrorString(status)});
    }
}

struct TMatrix {
    TMatrix(size_t cols, size_t rows, bool useLargeStride, half* storage)
        : Cols(cols)
        , Rows(rows)
        , Stride(useLargeStride ? LargeStride : Rows)
        , OwnsData(storage == nullptr)
    {
        if (OwnsData) {
            const auto sizeInFloats = Cols * Stride;
            const auto sizeInBytes = sizeInFloats * sizeof(half);
            CudaOrDie(cudaMalloc(&Ptr, sizeInBytes));
            CudaOrDie(cudaMemset(Ptr, 0, sizeInBytes));

            const auto start = uintptr_t(Ptr);
            const auto end = uintptr_t(Ptr + sizeInFloats);
            std::cout << "Allocated a segment: [" << std::hex << start << "; " << end << ")" << std::endl;
        } else {
            Ptr = storage;
        }
    }

    ~TMatrix() {
        if (OwnsData) {
            CudaOrDie(cudaFree(Ptr));
        }
    }

    size_t Cols;
    size_t Rows;
    size_t Stride;

    bool OwnsData;
    half* Ptr;
};

const half Alpha = 1.0f;
const half Beta = 0.0f;

class TCublasGemm {
public:
    TCublasGemm(TMatrix* a, TMatrix* b, TMatrix* c)
        : A(a)
        , B(b)
        , C(c)
    {
        CublasOrDie(cublasCreate(&Cbl));
    }

    ~TCublasGemm() {
        CublasOrDie(cublasDestroy(Cbl));
    }

    void Run() {
        CublasOrDie(
            cublasGemmEx(
                Cbl,
                CUBLAS_OP_N, CUBLAS_OP_N,
                A->Rows, B->Cols, A->Cols,
                &Alpha,
                A->Ptr, CUDA_R_16F, A->Stride,
                B->Ptr, CUDA_R_16F, B->Stride,
                &Beta,
                C->Ptr, CUDA_R_16F, C->Stride,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            )
        );
    }

private:
    TMatrix* A;
    TMatrix* B;
    TMatrix* C;

    cublasHandle_t Cbl;
};

int main(int argc, char** argv) {
    int cudaVersion{};
    CudaOrDie(cudaRuntimeGetVersion(&cudaVersion));
    const auto cudaMajorVersion = cudaVersion / 1000;

    if (argc != 2) {
        std::cerr << "Usage: <bin> overlap|no_overlap" << std::endl;
        return 1;
    }

    bool shouldOverlap{};
    if (strcmp(argv[1], "overlap") == 0) {
        shouldOverlap = true;
    } else if (strcmp(argv[1], "no_overlap") == 0) {
        shouldOverlap = false;
    } else {
        std::cerr << "Expected overlap or no_overlap as the only argument" << std::endl;
        return 1;
    }

    const bool isCuda11OrNewer = cudaMajorVersion >= 11;
    auto a = std::make_unique<TMatrix>(
        // A slightly different memory layout is required to trigger a crash in CUDA < 11.x.
        isCuda11OrNewer ? 3072 : 1024 /*cols*/,
        isCuda11OrNewer ? 1024 : 3072 /*rows*/,
        false /*useLargeStride*/,
        nullptr /*storage*/
    );

    auto b = std::make_unique<TMatrix>(
        100 /*cols*/,
        a->Cols /*rows*/,
        true /*useLargeStride*/,
        nullptr /*storage*/
    );

    auto c = std::make_unique<TMatrix>(
        b->Cols /*cols*/,
        a->Rows /*rows*/,
        true /*useLargeStride*/,
        shouldOverlap ? (b->Ptr + b->Rows) : nullptr /*storage*/
    );

    TCublasGemm gemm{a.get(), b.get(), c.get()};
    gemm.Run();

    CudaOrDie(cudaDeviceSynchronize());
}
