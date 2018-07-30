/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: MATRIX
************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

template<typename T>
matrix<T>::matrix(int x, int y)
    : _x(x), _y(y)
{
    //printf("Normal Construct\n");
    _data = new T[_x*_y];
}

template<typename T>
matrix<T>::matrix(const matrix& b) // copy construct
    : _x(b._x), _y(b._y), _data(new T[b._x*b._y])
{
    //printf("Copy Construct\n");
    memcpy(_data, b._data, sizeof(T)*_x*_y);
}

template<typename T>
matrix<T>& matrix<T>::operator=(const matrix& b) // copy assign
{
    //printf("Copy Assign\n");
    if (this != &b)
    {
        if (_x!=b._x || _y!=b._y)
        {
            _x = b._x;
            _y = b._y;
            delete[] _data;
            _data = new T[_x*_y];
        }
        memcpy(_data, b._data, sizeof(T)*_x*_y);
    }
    return *this;
}

template<typename T>
matrix<T>::matrix(matrix&& b) // move construct
    : _x(b._x), _y(b._y), _data(b._data)
{
    //printf("Move Construct\n");
    b._data = NULL;
    b._x = 0;
    b._y = 0;
}

template<typename T>
matrix<T>& matrix<T>::operator=(matrix&& b) // move assign
{
    //printf("Move Assign\n");
    if (this != &b)
    {
        delete[] _data;

        _x = b._x;
        _y = b._y;
        _data = b._data;

        b._x = 0;
        b._y = 0;
        b._data = NULL;
    }
    return *this;
}

template<typename T>
matrix<T>::~matrix()
{
    //printf("Delete\n");
    delete[] _data;
}

template<typename T>
void matrix<T>::display()
{
    for (int i=0;i<_x;i++)
    {
        for (int j=0;j<_y;j++)
            printf("%10d", _data[index(i, j)]);
        printf("\n");
    }
}

template<typename T>
void matrix<T>::reset(T value)
{
    for (int i=0;i<_x;i++)
    for (int j=0;j<_y;j++)
    _data[index(i, j)] = value;
}

template<typename T>
void matrix<T>::reset_num()
{
    int count = 1;
    for (int i=0;i<_x;i++)
    for (int j=0;j<_y;j++)
    _data[index(i, j)] = count++;
}

template<typename T>
matrix<T> matrix<T>::operator+(const matrix<T>& b)
{
    if (_x != b._x || _y != b._y)
    {
        printf("Shape Error!");
        return NULL;
    }

    matrix<T> c(_x, _y);
    for (int i=0;i<_x;i++)
    for (int j=0;j<_y;j++)
    c[i][j] = _data[index(i, j)] + b[i][j];

    return c;
}

template<typename T>
matrix<T> matrix<T>::operator*(const matrix<T>& b)
{
    if (_y != b._x)
    {
        printf("Shape Error!");
        return NULL;
    }

    matrix<T> c(_x, b._y);
    for (int i=0;i<_x;i++)
    for (int j=0;j<b._y;j++)
    {
        T temp = 0;
        for (int k=0;k<_y;k++)
        temp += _data[index(i, k)]*b[k][j];
        c[i][j] = temp;
    }

    //printf("--- Ready to exit ---\n");
    return c;
}

template<typename T>
T* matrix<T>::operator[](int x)
{
    return &_data[x*_x];
}

template<typename T>
const T* matrix<T>::operator[](int x) const
{
    return &_data[x*_x];
}

template class matrix<int>; // ! Important

#ifdef GPU_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#define ADDR(x) ((void**)&(x))

template<typename T>
gmatrix<T>::gmatrix(int x, int y)
    : matrix<T>(x, y)
{
    cudaMalloc(ADDR(_gpu_data), sizeof(T)*x*y);
}

template<typename T>
gmatrix<T>::gmatrix(const gmatrix& b) // copy construct
{
    matrix<T>::_x = b._x;
    matrix<T>::_y = b._y;
    matrix<T>::_data = new T[b._x*b._y];
    int data_size = sizeof(T)*matrix<T>::_x*matrix<T>::_y;
    memcpy(matrix<T>::_data, b._data, data_size);

    cudaMalloc(ADDR(_gpu_data), data_size);
    cudaMemcpy(_gpu_data, b._gpu_data, data_size, cudaMemcpyDeviceToDevice);
}

template<typename T>
gmatrix<T>& gmatrix<T>::operator=(const gmatrix& b) // copy assign
{
    if (this != &b)
    {
        int data_size = sizeof(T)*b._x*b._y;
        if (matrix<T>::_x!=b._x || matrix<T>::_y!=b._y)
        {
            matrix<T>::_x = b._x;
            matrix<T>::_y = b._y;
            delete[] matrix<T>::_data;
            matrix<T>::_data = new T[matrix<T>::_x*matrix<T>::_y];
            cudaFree(_gpu_data);
            cudaMalloc(ADDR(_gpu_data), data_size);
        }
        memcpy(matrix<T>::_data, b._data, data_size);
        cudaMemcpy(_gpu_data, b._gpu_data, data_size, cudaMemcpyDeviceToDevice);
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
        delete[] matrix<T>::_data;
        cudaFree(_gpu_data);

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
    cudaFree(_gpu_data);
}

#include "gemm.h"

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

template<typename T>
void gmatrix<T>::hTod()
{
    cudaMemcpy(_gpu_data, matrix<T>::_data,
        sizeof(T)*matrix<T>::_x*matrix<T>::_y,
        cudaMemcpyHostToDevice);
}

template<typename T>
void gmatrix<T>::dToh()
{
    cudaMemcpy(matrix<T>::_data, _gpu_data,
        sizeof(T)*matrix<T>::_x*matrix<T>::_y,
        cudaMemcpyDeviceToHost);
}

template class gmatrix<int>;

#endif // GPU_CUDA