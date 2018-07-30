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

using namespace std;

#define LAX 1024
#define LAY 1024
#define LBX 1024
#define LBY 1024

int main()
{
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
    
    return 0;
}