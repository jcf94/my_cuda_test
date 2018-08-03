/* ***********************************************
MYID	: Chen Fan
LANG	: G++
PROG	: GEMM
************************************************ */

template<typename T>
__global__ void add_kernel(T* dest, const T* a, const T* b, int x, int y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < x*y) dest[index] = a[index] + b[index];
}

#define TILE_WIDTH 16

template<typename T>
__global__ void mul_kernel_naive(T* c, const T* a, const T* b, int xx, int extra, int yy)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //printf("%d %d\n", x, y);
    T res = 0;
    for (int i=0;i<extra;i++)
        res += a[x*extra+i] * b[i*yy+y];

    if (x<xx && y<yy) c[x*yy+y] = res;
}

template<typename T>
__global__ void mul_kernel(T* c, const T* a, const T* b, int xx, int extra, int yy)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y; // Each thread calculate a number of final matrix
    //printf("%d %d\n", x, y);
    __shared__ T tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T tile_b[TILE_WIDTH][TILE_WIDTH]; // Cut A/B to 16*16
    int tile_num = 1 + (extra + TILE_WIDTH - 1) / TILE_WIDTH;
    T res = 0;
    for (int i=0;i<tile_num;i++) // Each move step is 16
    {
        int a_x = x;
        int a_y = i * TILE_WIDTH + tid_y;
        int b_x = i * TILE_WIDTH + tid_x;
        int b_y = y;
        // Copy to shared memory
        if (a_x < xx && a_y < extra) tile_a[tid_x][tid_y] = a[a_x * extra + a_y];
        else tile_a[tid_x][tid_y] = 0;
        if (b_x < extra && b_y < yy) tile_b[tid_x][tid_y] = b[b_x * yy + b_y];
        else tile_b[tid_x][tid_y] = 0; // 16 * 16 thread copy data
        __syncthreads();
        for (int t=0;t<TILE_WIDTH;t++)
            res += tile_a[tid_x][t] * tile_b[t][tid_y]; // add to result
        __syncthreads();
    }
    if (x<xx && y<yy) c[x*yy+y] = res;
}
