#ifndef __COMMON_CUDA_HPP__
#define __COMMON_CUDA_HPP__

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define HANDLE_ERROR(_exp)                                             \
    do                                                                 \
    {                                                                  \
        const cudaError_t err = (_exp);                                \
        if (err != cudaSuccess)                                        \
        {                                                              \
            std::cerr << cudaGetErrorString(err) << " in " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#endif // __COMMON_CUDA_HPP__
