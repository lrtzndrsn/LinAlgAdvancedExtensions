"""
@Project: LinalgDat 2022
@File: TestProjectC.py

@Description: Routines for testing implementations for the third LinalgDat
programming project.

A few routines are there to check validity of some vectors/matrix objects.
The tests have the same structure:
1) Check that the implemented function runs
2) When applicable, test that the result has the expected size/dimension
3) Check that the result has the expected value
Only when all these checks are successful does the test method return True.

This an adaptation of my previous F# and C# code
"""
import math
from sys import (
    path,
    modules
)
path.append('../Core')
from Vector import Vector
from Matrix import Matrix

import AdvancedExtensions
from data_projectC import (
    SSMArgs,
    DeterminantArgs,
    SCArgs,
    GSArgs
)

__author__ = "François Lauze, University of Copenhagen"
__date__ = "04/28/22"
__version__ = "0.0.1"

Tolerance = 1e-3 # keep it large because of rounding/precision error when emitting data
GWidth = 60
TopString = '\n' + ''.join(['='] * (GWidth + 9))
BotString = '\n' + ''.join(['-'] * (GWidth + 9)) + '\n'


class ProjectBException(Exception):
    """Dummy exception class for the project."""
    pass


def l1_distance(v: Vector, w: Vector) -> float:
    """
    l1-distance between two vectors of the same length.

    Internal use, as the lengths are actually not compared.
    """
    l1 = 0.0
    for i in range(len(v)):
        l1 += abs(v[i] - w[i])
    return l1


def compareVectorDimensions(v: Vector, w: Vector) -> bool:
    """Return True if v and w have same length, False otherwise."""
    return len(v) == len(w)


def compareVectors(v: Vector, w: Vector) -> bool:
    """
    Return True if they have the same length and L1-distance is less than Tolerance.

    Somewhat obfuscated version...
    """
    if compareVectorDimensions(v, w):
        return l1_distance(v, w) < Tolerance
    else:
        return False


def compareMatrixDimensions(A: Matrix, B: Matrix) -> bool:
    """Compare matrix dimensions."""
    return (A.N_Cols == B.N_Cols) and (A.M_Rows == B.M_Rows)


def compareMatrices(A: Matrix, B: Matrix) -> bool:
    """Compare matrices up to tolerance via l1-distance."""
    if compareMatrixDimensions(A, B):
        l1_norm = 0.0
        for i in range(A.N_Cols):
            l1_norm += l1_distance(A.Row(i), B.Row(i))
        return l1_norm < Tolerance
    else:
        raise ProjectBException("Matrix dimensions mismatch.")


def outMessage(taskName: str, subTaskName: str, status: bool) -> str:
    """Display a message followed by [PASSED] or [FAILED]."""
    passed = '[PASSED]'
    failed = '[FAILED]'
    s = f'{taskName} {subTaskName}'
    return f'{s:{GWidth}} {passed if status else failed}'


# noinspection PyBroadException
def TestSSM(A: Matrix, p: tuple, expected: Matrix) -> tuple:
    """Test that SquareSubMatrix returns a correct result."""
    taskName = 'SquareSubMatrix(Matrix, int, int)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += TopString
    i, j = p
    try:
        B = AdvancedExtensions.SquareSubMatrix(A, i, j)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += BotString
    return taskName, status, resultStr


def TestDet(A: Matrix, expected: float) -> tuple:
    """Test that Determinant returns a correct result."""
    taskName = 'Determinant(Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += TopString
    try:
        det = AdvancedExtensions.Determinant(A)
        try:
            if not math.isclose(det, expected, abs_tol=Tolerance):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += BotString
    return taskName, status, resultStr


def TestSC(A: Matrix, v: Vector, i: int, expected: Matrix) -> tuple:
    """Test that SetColumn returns a correct result."""
    taskName = 'SetColumn(Matrix, Vector, int)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += TopString
    try:
        B = AdvancedExtensions.SetColumn(A, v, i)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += BotString
    return taskName, status, resultStr


def TestGS(A: Matrix, expected: tuple) -> tuple:
    """Test that Gram-Schmidt returns a correct result."""
    taskName = 'GramSchmidt(Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += TopString
    Qe, Re = expected
    try:
        Q, R = AdvancedExtensions.GramSchmidt(A)
        try:
            if (not compareMatrices(Q, Qe)) or (not compareMatrices(R, Re)):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += BotString
    return taskName, status, resultStr


def runTestSSM(matrices: list, indices: list, expected_matrices: list) -> tuple:
    """Run Square submatrix tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, p, E in zip(matrices, indices, expected_matrices):
        taskName, status, resultString = TestSSM(A, p, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestDet(matrices: list,  expected_values: list) -> tuple:
    """Run Determinant tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, d in zip(matrices, expected_values):
        taskName, status, resultString = TestDet(A, d)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestSC(matrices: list, vectors: list, indices: list, expected_matrices: list) -> tuple:
    """Run SetColumn tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, v, i, E in zip(matrices, vectors, indices,  expected_matrices):
        taskName, status, resultString = TestSC(A, v, i, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestGS(matrices: list, expected_matrices: list) -> tuple:
    """Run forward reduction tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A,  QR in zip(matrices, expected_matrices):
        taskName, status, resultString = TestGS(A, QR)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def printSummaryInfo(taskName: str, passed: int, total: int):
    """Print a line with execution summary for a given task."""
    str1 = f'Test of {taskName} passed/total'
    print(f'{str1:<70} [{passed}/{total}]')


def runALL():
    results = [
        runTestSSM(*SSMArgs),
        runTestDet(*DeterminantArgs),
        runTestSC(*SCArgs),
        runTestGS(*GSArgs)
    ]

    print('=' * 80)
    for result in results:
        printSummaryInfo(*result)
    print('-' * 80)


if __name__ == '__main__':
    if 'numpy' in modules:
        print("Numpy was imported!!!!!!!")
    else:
        runALL()