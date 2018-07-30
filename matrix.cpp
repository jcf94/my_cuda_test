/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: MATRIX
************************************************ */

#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"

template<typename T>
matrix<T>::matrix(int x, int y)
{
    //printf("Normal Construct\n");
    _x = x;
    _y = y;
    _data = new T[_x*_y];
}

template<typename T>
matrix<T>::matrix(const matrix& b) // copy construct
    : _x(b._x), _y(b._y), _data(new T[b._x*b._y])
{
    //printf("Copy Construct\n");
    for (int i=0;i<_x;i++)
    for (int j=0;j<_y;j++)
    _data[index(i, j)] = b[i][j];
}

template<typename T>
matrix<T>& matrix<T>::operator=(const matrix& b) // copy assign
{
    //printf("Copy Assign\n");
    if (this != &b)
    {
        delete[] _data;

        _x = b._x;
        _y = b._y;
        _data = new T[_x*_y];

        for (int i=0;i<_x;i++)
        for (int j=0;j<_y;j++)
        _data[index(i, j)] = b[i][j];
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