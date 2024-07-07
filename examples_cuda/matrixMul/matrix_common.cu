/* ************************************************************************
 * Description: Common routines for matrix multiplication
 **************************************************************************/
#include <stdio.h>
#include "matrix_common.h"

#define DEFAULT_CTE 101

/* ************************************************************************
 * Routine: Initialize
 * Description: Initializes all elements in matrix.
 **************************************************************************/
void Initialize(float *A, int fil, int col) {
  int i, j;

  for(i = 0; i < fil; i++)
    for(j = 0; j < col; j++)
      *(A + i * col + j) = rand() % DEFAULT_CTE;
}

/* ************************************************************************
 * Routine: InitializeI
 * Description: Initializes each square submatrix in matrix A to the
 *   identity I.
 **************************************************************************/
void InitializeI(float *A, int fil, int col) {
  int i, j;

  for(i = 0; i < fil; i++)
    for(j = 0; j < col; j++) 
      *(A + i * col + j) = ((i % col) == (j % fil)) ? 1.0 : 0.0;
}

/* ************************************************************************
 * Routine: display
 * Description: Displays a matrix A to stdout.
 **************************************************************************/
void display(float *A, int fil, int col) {
  int i, j;

  for(i = 0; i < fil; i++) {
    for(j = 0; j < col; j++)
      printf("%9.2f ", *(A + i * col + j)); 
    printf("\n");
  }
  printf("=====================================\n");
}

/* ************************************************************************
 * Routine: Compare
 * Description: Returns 1 if matrices are identical
 **************************************************************************/
int Compare(float *A, float *B, int fil, int col) {
  int i, j;

  for(i = 0; i < fil; i++)
    for(j = 0; j < col; j++)
      if(*(A + i * col + j) != *(B + i * col + j))
        return 0;
  return 1;
}

/* ************************************************************************
 * Routine: matmul_seq 
 * Description: Matrix multiplication
 **************************************************************************/
void matmul_seq(float *C, float *A, float *B, int N) {
  int i, j, k;

  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++)
      for(C[i*N+j] = 0.0, k = 0; k < N; k++) 
        C[i*N+j] += A[i*N+k] * B[k*N+j];
}

