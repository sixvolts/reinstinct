// Smoke-test kernel: y[i] = a[i] + b[i].
//
// Verifies the entire HIP toolchain end-to-end: hipcc → .hsaco compile,
// hipModuleLoad, hipModuleGetFunction, hipModuleLaunchKernel, and result
// equality vs a CPU reference. Not a perf kernel — block size / grid stride
// are deliberately simple.

#include <hip/hip_runtime.h>

extern "C" __global__
void vector_add_f32(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float*       __restrict__ y,
                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}
