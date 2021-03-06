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
    inline int index(int i, int j) const
    {
        return i*_y + j;
    }
    inline int x() const
    {
        return _x;
    }
    inline int y() const
    {
        return _y;
    }
    inline T* data() const
    {
        return _data;
    }
    T* operator[](int x);
    const T* operator[](int x) const;

    // Data Output & Input
    void display() const;
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

#endif // !MATRIX_H