/* ************************************************************************
 * Description: Common routines for matrix multiplication
 **************************************************************************/

#ifndef _MATRIX_COMMON_H_
#define _MATRIX_COMMON_H_

/* ************************************************************************
 * Routine: Initialize
 * Description: Initializes all elements in matrix.
 **************************************************************************/
void Initialize(float *A, int fil, int col);

/* ************************************************************************
 * Routine: InitializeI
 * Description: Initializes each square submatrix in matrix A to the
 *   identity I.
 **************************************************************************/
void InitializeI(float *A, int fil, int col);

/* ************************************************************************
 * Routine: display
 * Description: Displays a matrix A to stdout.
 **************************************************************************/
void display(float *A, int fil, int col);

/* ************************************************************************
 * Routine: Compare
 * Description: Returns 1 if matrices are identical
 **************************************************************************/
int Compare(float *A, float *B, int fil, int col);

/* ************************************************************************
 * Routine: matmul_seq
 * Description: Matrix multiplication
 **************************************************************************/
void matmul_seq(float *C, float *A, float *B, int N);

#endif
