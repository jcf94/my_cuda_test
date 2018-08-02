/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: Main
************************************************ */

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <omp.h>

#ifdef GPU_CUDA
#include "cuda_test.h"
#include "gmatrix.h"
#else
#include "matrix.h"
#endif // GPU_CUDA

using namespace std;

#define LAX 4
#define LAB 6
#define LBY 4

int main()
{
    // printf("CPU Matrix Test\n");
    // printf("---------------\n");
    // matrix<int> a;
    // a.reset_num();
    // a.display();

    // printf("---------------\n");
    // matrix<int> b;
    // b.reset_num();
    // b.display();

    // printf("---------------\n");
    // matrix<int> c;
    // c = a*b;
    // c.display();

    #ifdef GPU_CUDA
    check_device();

    printf("GPU Matrix Test\n");

    printf("---------------\n");
    gmatrix<int> ga(LAX, LAB);
    ga.reset_num();
    ga.display();
    ga.hTod();

    printf("---------------\n");
    gmatrix<int> gb(LAB, LBY);
    gb.reset_num();
    gb.display();
    gb.hTod();

    // printf("---------------\n");
    // gmatrix<int> gc;
    // double start, end;
    // start = omp_get_wtime();
    // gc = ga*gb;
    // end = omp_get_wtime();
    // printf("GPU cost: %.2lf ms.\n", end - start);
    // gc.dToh();

    // printf("---------------\n");
    // gmatrix<int> gd;
    // matrix<int> d(LAB, LBY);
    // d.reset_num();
    // start = omp_get_wtime();
    // gd = ga*d;
    // end = omp_get_wtime();
    // printf("CPU cost: %.2lf s.\n", end - start);

    // if (gd.equal(gc)) printf("PASS\n");
    // else printf("ERROR!\n");

    #endif // GPU_CUDA

    return 0;
}