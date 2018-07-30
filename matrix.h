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

    void display();

    void reset(T value = 0);
    void reset_num();

    matrix operator+(const matrix& b);
    matrix operator*(const matrix& b);

private:
    T* _data;
    int _x;
    int _y;
};

#endif // !MATRIX_H