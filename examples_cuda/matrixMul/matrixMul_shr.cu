#include <stdio.h>
#include "common.h"
#include "matrix_common.h"

#define BLOCK_SIZE 16
#define DEFAULT_N 1

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul(float* C, float* A, float* B, int N) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = N * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + N - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * N;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + N * ty + tx];
    Bs[ty][tx] = B[b + N * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    // #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
      Csub += As[ty][k] * Bs[k][tx];

     // Synchronize to make sure that the preceding
     // computation is done before loading two new
     // sub-matrices of A and B in the next iteration
     __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + N * ty + tx] = Csub;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  int N, sizeinBytes;
  float *A, *B, *C, *D;
  float *d_A, *d_B, *d_C;

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

  // Timing
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

  // setup execution parameters
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(N, N);

  // execute the kernel
  matrixMul<<< dimGrid, dimBlock >>>(d_C, d_A, d_B, N*BLOCK_SIZE);

  // Results moving
  HANDLE_ERROR( cudaMemcpy(C, d_C, sizeinBytes, cudaMemcpyDeviceToHost) );

  // Timing
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  matmul_seq(D, A, B, N*BLOCK_SIZE);

  printf("%s: N = %4d Test = %d Time = %3.1f ms\n", argv[0], N*BLOCK_SIZE, \
         Compare(D, C, N*BLOCK_SIZE, N*BLOCK_SIZE), elapsedTime);

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

