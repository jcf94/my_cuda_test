/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: CUDA_TEST
************************************************ */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_test.h"

void check_device()
{
    printf("Starting...\n\n");
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
}