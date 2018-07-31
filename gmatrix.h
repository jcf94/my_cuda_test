/* ***********************************************
MYID	: Chen Fan
LANG	: NVCC
PROG	: GMATRIX_H
************************************************ */

#ifndef GMATRIX_H
#define GMATRIX_H

#include "matrix.h"

template<typename T>
class gmatrix : public matrix<T>
{
public:
    // Constructor & Destructor
    gmatrix(int x = DEFAULT_X, int y = DEFAULT_Y);
    gmatrix(const gmatrix& b); // copy construct
    gmatrix(gmatrix&& b); // move construct
    gmatrix& operator=(const gmatrix& b); // copy assign
    gmatrix& operator=(gmatrix&& b); // move assign
    ~gmatrix();

    // Data index
    inline int index(int x, int y)
    {
        return x*matrix<T>::_x + y;
    }
    inline int x()
    {
        return matrix<T>::_x;
    }
    inline int y()
    {
        return matrix<T>::_y;
    }
    inline T* data()
    {
        return matrix<T>::_data;
    }

    // Data Output & Input
    void hTod();
    void dToh();

    // Calculate
    gmatrix operator+(const gmatrix& b);
    gmatrix operator*(const gmatrix& b);
    gmatrix operator*(matrix<T> b);

    bool equal(gmatrix b);

private:
    T* _gpu_data;
};

#endif // !GMATRIX_H