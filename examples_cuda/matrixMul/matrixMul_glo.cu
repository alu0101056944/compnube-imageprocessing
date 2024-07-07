/* ************************************************************************
 * Program: matrix.c
 * Description: Matrix multiplication
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "matrix_common.h"

#define DEFAULT_N   1
#define BLOCK_SIZE 16

/* ************************************************************************
 * Routine: matmul_kernel 
 * Description: Matrix multiplication
 **************************************************************************/
__global__ void matmul_kernel(float *C, float *A, float *B, int N) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if((i<N) && (j<N))
  {
    C[i*N+j] = 0;
    for(int k = 0; k < N; k++)
      C[i*N+j] += A[i*N+k] * B[k*N+j];
  }
}

/* ************************************************************************
 * Routine: main
 * Description: Performs several tests.
 **************************************************************************/
int main(int argc, char *argv[]) {
  int N, sizeinBytes;
  float *A, *B, *C, *D;
  float *d_A, *d_B, *d_C;

  // Timers
  cudaEvent_t start, stop;

  /* Command line parameters processing */
  switch(argc) {
    case 1: N = DEFAULT_N;
            break;
    case 2: N = atoi(argv[1]);
            break;
    default: 
            printf("\nUse: %s <N>", argv[0]);
            printf("\nN: Dimensión del GRID en bloques de %d)\n", BLOCK_SIZE);
            break;
  }

  sizeinBytes = N*N*BLOCK_SIZE*BLOCK_SIZE*sizeof(float);

  A = (float*)malloc(sizeinBytes);
  B = (float*)malloc(sizeinBytes);
  C = (float*)malloc(sizeinBytes);
  D = (float*)malloc(sizeinBytes);
 
  HANDLE_NULL(A);
  HANDLE_NULL(B);
  HANDLE_NULL(C);
  HANDLE_NULL(D);

  // Time
  HANDLE_ERROR( cudaEventCreate( &start ) );
  HANDLE_ERROR( cudaEventCreate( &stop ) );
  HANDLE_ERROR( cudaEventRecord( start, 0 ) );

  // Memory allocation
  HANDLE_ERROR( cudaMalloc(&d_A, sizeinBytes) );
  HANDLE_ERROR( cudaMalloc(&d_B, sizeinBytes) );
  HANDLE_ERROR( cudaMalloc(&d_C, sizeinBytes) );

  // Host initializing
  Initialize(A, N*BLOCK_SIZE, N*BLOCK_SIZE);
  Initialize(B, N*BLOCK_SIZE, N*BLOCK_SIZE);

  // Device initializing
  HANDLE_ERROR( cudaMemcpy(d_A, A, sizeinBytes, cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_B, B, sizeinBytes, cudaMemcpyHostToDevice) );

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(N, N);

  matmul_kernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N*BLOCK_SIZE);

  HANDLE_ERROR( cudaMemcpy(C, d_C, sizeinBytes, cudaMemcpyDeviceToHost) );
  
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );

  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  matmul_seq(D, A, B, N*BLOCK_SIZE);

  printf("%s: N = %4d Test = %d Time = %3.1f ms\n", argv[0], N*BLOCK_SIZE, \
         Compare(D, C, N*BLOCK_SIZE, N*BLOCK_SIZE), elapsedTime);

#ifdef DEBUG
  display(A, N*BLOCK_SIZE, N*BLOCK_SIZE);
  display(B, N*BLOCK_SIZE, N*BLOCK_SIZE);
  display(C, N*BLOCK_SIZE, N*BLOCK_SIZE);
#endif

  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );

  HANDLE_ERROR ( cudaFree(d_A) );
  HANDLE_ERROR ( cudaFree(d_B) );
  HANDLE_ERROR ( cudaFree(d_C) );

  free(A);
  free(B);
  free(C);
  free(D);

  return 0;
}

