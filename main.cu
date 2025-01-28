#include <iostream>
#include <math.h>

#include "api.h"

__kernel__ void add(int n, float* x, float* y, float* z)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        z[i] = x[i] + y[i];
    }
}

int main(void)
{
    int N = 1 << 20; // 1,048,576

    float *x, *y, *z;
    float *d_x, *d_y, *d_z;

    // Allocate memory on host (CPU)
    x = new float[N];
    y = new float[N];
    z = new float[N];

    // Allocate memory on device (GPU)
    checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_z, N * sizeof(float)));

    // Initialize x and y arrays on the host (CPU)
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0;
    }

    // Copy x and y arrays from host to device
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256; // More optimal thread count for modern GPUs
    int blocks = (N + threads - 1) / threads;

    std::cout << blocks << ", " << threads << std::endl;

    // Run kernel on 1M elements on the GPU
    add<<<blocks, threads>>>(N, d_x, d_y, d_z);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check for errors (all values should be 3.0f)
    float max_error = 0.0f;
    for (int i = 0; i < N; i++)
    {
        max_error = std::fmax(max_error, std::fabs(z[i] - 3.0f));
    }

    std::cout << "Max error: " << max_error << std::endl;

    // Free memory on device (GPU)
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));

    // Free memory on host (CPU)
    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}
