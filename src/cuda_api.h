#pragma once

#ifdef __CUDACC__
#define __kernel__ __global__
#define __cpu__ __host__
#define __gpu__ __device__
#define __cpu_gpu__ __host__ __device__
#else
#define __kernel__
#define __cpu__
#define __gpu__
#define __cpu_gpu__
#endif
