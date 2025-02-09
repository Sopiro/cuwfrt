#pragma once

#ifdef __CUDACC__
#define __KERNEL__ __global__
#define __CPU__ __host__
#define __GPU__ __device__
#define __CPU_GPU__ __host__ __device__
#else
#define __KERNEL__
#define __CPU__
#define __GPU__
#define __CPU_GPU__
#endif
