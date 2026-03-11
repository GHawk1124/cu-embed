extern "C" __global__ void add_scalar(float* values, float scalar, int len) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        values[idx] += scalar;
    }
}

