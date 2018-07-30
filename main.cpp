/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: Main
************************************************ */

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "matrix.h"

#ifdef GPU_CUDA
#include "cuda_test.h"
#endif // GPU_CUDA

using namespace std;

#define LAX 1024
#define LAY 1024
#define LBX 1024
#define LBY 1024

int main()
{
    printf("CPU Matrix Test\n");
    printf("---------------\n");
    matrix<int> a;
    a.reset_num();
    a.display();

    printf("---------------\n");
    matrix<int> b;
    b.reset_num();
    b.display();

    printf("---------------\n");
    matrix<int> c;
    c = a*b;
    c.display();

    #ifdef GPU_CUDA
    check_device();

    printf("GPU Matrix Test\n");

    printf("---------------\n");
    gmatrix<int> ga;
    ga.reset_num();
    ga.display();
    ga.hTod();

    printf("---------------\n");
    gmatrix<int> gb;
    gb.reset_num();
    gb.display();
    gb.hTod();

    printf("---------------\n");
    gmatrix<int> gc;
    gc = ga*gb;
    gc.dToh();
    gc.display();

    #endif // GPU_CUDA

    return 0;
}