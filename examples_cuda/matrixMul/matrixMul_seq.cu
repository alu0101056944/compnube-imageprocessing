/* ************************************************************************
 * Program: matrix.c
 * Description: Matrix multiplication
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "matrix_common.h"

#define DEFAULT_N   16

/* ************************************************************************
 * Routine: main
 * Description: Performs several tests.
 **************************************************************************/
int main(int argc, char *argv[]) {
  int N;
  float *A, *B, *C;

  /* Command line parameters processing */
  switch(argc) {
    case 1: N = DEFAULT_N;
            break;
    case 2: N = atoi(argv[1]);
            break;
    default: 
            printf("\nUse: %s <N>", argv[0]);
            printf("\nN: Dimension de la matriz)\n");
            break;
  }

  A = (float*)calloc(N * N, sizeof(float));
  B = (float*)calloc(N * N, sizeof(float));
  C = (float*)calloc(N * N, sizeof(float));
  if(!A || !B || !C) {
    printf("\nError: No memory enough\n");
    exit(-1);
  }
 
  Initialize(A, N, N);
  InitializeI(B, N, N);

  matmul_seq(C, A, B, N);

  printf("%s: N = %4d Test = %d\n", argv[0], N, Compare(A, C, N, N));

#ifdef DEBUG
  display(A, N, N);
  display(B, N, N);
  display(C, N, N);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
