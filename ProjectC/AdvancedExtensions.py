# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: AdvancedExtensions.py

@Description: Project C Determinant and Gram-Schmidt extensions.

"""

import math
import sys

sys.path.append('../Core')
from Vector import Vector
from Matrix import Matrix

Tolerance = 1e-6


def SquareSubMatrix(A: Matrix, i: int, j: int) -> Matrix:
    """
    This function creates the square submatrix given a square matrix as
    well as row and column indices to remove from it.

    Remarks:
        See page 246-247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameters:
        A:  N-by-N matrix
        i: int. The index of the row to remove.
        j: int. The index of the column to remove.

    Return:
        The resulting (N - 1)-by-(N - 1) submatrix.
    """
     
    N = A.N_Cols  
    B = Matrix(N-1, N-1)
    ARow = 0
    for r in range(N):  
        if r != i: 
            ACol = 0
            for c in range(N):  
                if c != j: 
                    B[ARow, ACol] = A[r,c]
                    ACol = ACol + 1
            ARow = ARow + 1 
    return B 


def Determinant(A: Matrix) -> float:
    """
    This function computes the determinant of a given square matrix.

    Remarks:
        * See page 247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.
        * Hint: Use SquareSubMatrix.

    Parameter:
        A: N-by-N matrix.

    Return:
        The determinant of the matrix.
    """
    M = A.N_Cols
    if M == 2:
      detSizeTwoMatrix = (A[0,0] * A[1,1]) - (A[0,1] * A[1,0])
      return detSizeTwoMatrix 
    else: 
        det = 0
        N = M 
        for j in range (N):
            cofactor = (-1) ** (0 + j) * A[0,j] * Determinant(SquareSubMatrix(A, 0, j))
            det += cofactor 
        return det 
        
    
def VectorNorm(v: Vector) -> float:
    """
    This function computes the Euclidean norm of a Vector. This has been implemented
    in Project A and is provided here for convenience

    Parameter:
         v: Vector

    Return:
         Euclidean norm, i.e. (\sum v[i]^2)^0.5
    """
    nv = 0.0
    for i in range(len(v)):
        nv += v[i]**2
    return math.sqrt(nv)


def SetColumn(A: Matrix, v: Vector, j: int) -> Matrix:
    """
    This function copies Vector 'v' as a column of Matrix 'A'
    at column position j.

    Parameters:
        A: M-by-N Matrix.
        v: size M vector
        j: int. Column number.

    Return:
        Matrix A  after modification.

    Raise:
        ValueError if j is out of range or if len(v) != A.M_Rows.
    """
    N = A.M_Rows  
    for i in range(N): 
         A[i,j] = v[i]    
    return A 
    #raise NotImplementedError()

def Transpose(A: Matrix) -> Matrix:
    M = A.M_Rows 
    N = A.N_Cols  
    C = Matrix(N, M)        
    for i in range(M): #for hver row
        for j in range (N): #for hver col
            C[j, i] = A[i,j]
    return C

def VectorScalar (v1: Vector, v2: Vector ) -> float: 
    vecscalar = 0 
    for i in range (len(v1)): 
        vecscalar += v1[i]*v2[i]
    return vecscalar 

def GramSchmidt(A: Matrix) -> tuple:
    """
    This function computes the Gram-Schmidt process on a given matrix.

    Remarks:
        See page 229 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameter:
        A: M-by-N matrix. All columns are implicitly assumed linear
        independent.

    Return:
        tuple (Q,R) where Q is a M-by-N orthonormal matrix and R is an
        N-by-N upper triangular matrix.
    """
    M = A.M_Rows
    N = A.N_Cols
    Q = Matrix(M, N)
    R = Matrix(N, N)
    for j in range(N):
        v = Matrix.Column(A,j)
        SetColumn(Q, v, j)
        for i in range (j):
            q = Matrix.Column(Q, i)
            R[i, j] = VectorScalar(q, v) 
            v1 = Matrix.Column(Q, j) - Vector.internalMul(q, R[i, j]) 
            SetColumn(Q, v1, j)
        if VectorNorm(Matrix.Column(Q,j)) > Tolerance :
            R[j, j] = VectorNorm(Matrix.Column(Q,j))
            v2 = Vector.internalMul(Matrix.Column(Q,j), 1 / R[j, j])
            SetColumn (Q, v2, j)
    return Q, R
