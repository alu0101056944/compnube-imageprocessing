// Suma de vectores con kernel en 1 dimensión

#include "common.h"

#define N   10

__global__ void add_block( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id

    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

__global__ void add_thread( int *a, int *b, int *c ) {
    int tid = threadIdx.x;    // this thread handles the data at its thread id

    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // this thread handles the data at its thread id

    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main( void ) {
    int a[N], b[N], c[N], d[N], e[N];
    int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e;

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_d, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_e, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

//----------------------------------------------------------------
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add_thread<<<1,N>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

//----------------------------------------------------------------
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add_block<<<N,1>>>( dev_a, dev_b, dev_d );

    // copy the array 'd' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( d, dev_d, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], d[i] );
    }

//----------------------------------------------------------------
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add<<<ceil(N/5.0),5>>>( dev_a, dev_b, dev_e );

    // copy the array 'd' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( e, dev_e, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], e[i] );
    }

//----------------------------------------------------------------

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );
    HANDLE_ERROR( cudaFree( dev_d ) );
    HANDLE_ERROR( cudaFree( dev_e ) );

    return 0;
}
