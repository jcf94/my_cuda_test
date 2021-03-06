/* ***********************************************
MYID	: Chen Fan
LANG	: NVCC
PROG	: GMATRIX
************************************************ */

#include "gmatrix.h"

#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

#define check_cuda_rt(x) do { cudaError_t error__ = x; if (error__ != cudaSuccess) { printf("Code: %s, Description: %s", cudaGetErrorName(error__), cudaGetErrorString(error__)); exit(EXIT_FAILURE); } } while(false)

#define ADDR(x) ((void**)&(x))

#include "gemm.h"

template<typename T>
gmatrix<T>::gmatrix(int x, int y)
    : matrix<T>(x, y)
{
    //printf("gmatrix Normal Construct\n");
    check_cuda_rt(cudaMalloc(ADDR(_gpu_data), sizeof(T)*x*y));
}

template<typename T>
gmatrix<T>::gmatrix(const gmatrix& b) // copy construct
{
    //printf("Copy Construct\n");

    matrix<T>::_x = b._x;
    matrix<T>::_y = b._y;
    int data_size = sizeof(T)*x()*y();

    matrix<T>::_data = (T*)malloc(data_size);
    memcpy(matrix<T>::_data, b.data(), data_size);

    check_cuda_rt(cudaMalloc(ADDR(_gpu_data), data_size));
    check_cuda_rt(cudaMemcpy(_gpu_data, b._gpu_data, data_size, cudaMemcpyDeviceToDevice));
}

template<typename T>
gmatrix<T>& gmatrix<T>::operator=(const gmatrix& b) // copy assign
{
    //printf("Copy Assign\n");

    if (this != &b)
    {
        int data_size = sizeof(T)*b._x*b._y;
        if (matrix<T>::_x!=b._x || matrix<T>::_y!=b._y)
        {
            matrix<T>::_x = b._x;
            matrix<T>::_y = b._y;
            free(matrix<T>::_data);
            matrix<T>::_data = (T*)malloc(data_size);
            check_cuda_rt(cudaFree(_gpu_data));
            check_cuda_rt(cudaMalloc(ADDR(_gpu_data), data_size));
        }
        memcpy(matrix<T>::_data, b._data, data_size);
        check_cuda_rt(cudaMemcpy(_gpu_data, b._gpu_data, data_size, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

template<typename T>
gmatrix<T>::gmatrix(gmatrix&& b) // move construct
{
    matrix<T>::_x = b._x;
    matrix<T>::_y = b._y;
    matrix<T>::_data = b._data;
    _gpu_data = b._gpu_data;

    b._data = NULL;
    b._gpu_data = NULL;
    b._x = 0;
    b._y = 0;
}

template<typename T>
gmatrix<T>& gmatrix<T>::operator=(gmatrix&& b) // move assign
{
    if (this != &b)
    {
        free(matrix<T>::_data);
        check_cuda_rt(cudaFree(_gpu_data));

        matrix<T>::_x = b._x;
        matrix<T>::_y = b._y;
        matrix<T>::_data = b._data;
        _gpu_data = b._gpu_data;

        b._x = 0;
        b._y = 0;
        b._data = NULL;
        b._gpu_data = NULL;
    }
    return *this;
}

template<typename T>
gmatrix<T>::~gmatrix()
{
    //printf("gmatrix Delete\n");
    check_cuda_rt(cudaFree(_gpu_data));
}

template<typename T>
gmatrix<T> gmatrix<T>::operator+(const gmatrix<T>& b)
{
    if (matrix<T>::_x != b._x || matrix<T>::_y != b._y)
    {
        printf("Shape Error!");
        return NULL;
    }

    gmatrix<T> c(matrix<T>::_x, matrix<T>::_y);

    int block_dim = 512;
    int grid_dim = (matrix<T>::_x*matrix<T>::_y + block_dim - 1) / block_dim;
    add_kernel<T><<<grid_dim, block_dim>>>(c._gpu_data, _gpu_data, b._gpu_data,
        matrix<T>::_x, matrix<T>::_y);
    cudaDeviceSynchronize();

    return c;
}

template<typename T>
gmatrix<T> gmatrix<T>::operator*(const gmatrix<T>& b)
{
    if (matrix<T>::_y != b._x)
    {
        printf("Shape Error!");
        return NULL;
    }

    gmatrix<T> c(matrix<T>::_x, b._y);

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim((matrix<T>::_x + TILE_WIDTH - 1) / TILE_WIDTH, (b._y + TILE_WIDTH - 1) / TILE_WIDTH);
    mul_kernel<T><<<grid_dim, block_dim>>>(c._gpu_data, _gpu_data, b._gpu_data,
        matrix<T>::_x, matrix<T>::_y, b._y);
    cudaDeviceSynchronize();

    return c;
}

#include <omp.h>

template<typename T>
gmatrix<T> gmatrix<T>::operator*(const matrix<T>& b)
{
    if (y() != b.x())
    {
        printf("Shape Error!");
        return NULL;
    }

    gmatrix<T> c(x(), b.y());
    for (int i=0;i<x();i++)
    #pragma omp parallel for
    for (int j=0;j<b.y();j++)
    {
        T temp = 0;
        for (int k=0;k<y();k++)
        temp += matrix<T>::_data[index(i, k)]*b[k][j];
        c[i][j] = temp;
    }

    return c;
}

template<typename T>
void gmatrix<T>::hTod()
{
    check_cuda_rt(cudaMemcpy(_gpu_data, data(),
        sizeof(T)*x()*y(),
        cudaMemcpyHostToDevice));
}

template<typename T>
void gmatrix<T>::dToh()
{
    check_cuda_rt(cudaMemcpy(data(), _gpu_data,
        sizeof(T)*x()*y(),
        cudaMemcpyDeviceToHost));
}

template<typename T>
bool gmatrix<T>::equal(gmatrix<T> b) const
{
    if (x()!=b.x() || y()!=b.y()) return false;
    for (int i=0;i<x();i++)
    for (int j=0;j<y();j++)
    if (data()[index(i, j)] != b[i][j]) return false;
    return true;
}

template class gmatrix<int>;
template class gmatrix<float>;
template class gmatrix<double>;