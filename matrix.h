/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: MATRIX_H
************************************************ */

#ifndef MATRIX_H
#define MATRIX_H

#define DEFAULT_X 4
#define DEFAULT_Y 4

template<typename T>
class matrix
{
public:
    // Constructor & Destructor
    matrix(int x = DEFAULT_X, int y = DEFAULT_Y);
    matrix(const matrix& b); // copy construct
    matrix(matrix&& b); // move construct
    matrix& operator=(const matrix& b); // copy assign
    matrix& operator=(matrix&& b); // move assign
    ~matrix();

    // Data index
    inline int index(int x, int y)
    {
        return x*_x + y;
    }
    T* operator[](int x);
    const T* operator[](int x) const;

    // Data Output & Input
    void display();
    void reset(T value = 0);
    void reset_num();

    // Calculate
    matrix operator+(const matrix& b);
    matrix operator*(const matrix& b);

protected:
    T* _data;
    int _x;
    int _y;
};

#ifdef GPU_CUDA

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

    // Data Output & Input
    void hTod();
    void dToh();

    // Calculate
    gmatrix operator+(const gmatrix& b);
    gmatrix operator*(const gmatrix& b);

private:
    T* _gpu_data;
};

#endif // GPU_CUDA

#endif // !MATRIX_H