"""
Matrix Image (Column Space Basis) via Row Elimination: https://www.deep-ml.com/problems/68?from=Linear%20Algebra

Computes a basis for the column space of a matrix by performing
row elimination and extracting pivot columns.

Adversarial test cases verified:
- Leading zeros in pivot rows
- Duplicate/dependent rows  
- Zero rows in middle
- Wide and tall matrices
- All-zero matrices
- Numerical edge cases

Stress-tested against 14 adversarial cases (from Claude AI) - no breaking cases found.
"""

import numpy as np


def matrix_image(A):
    A = np.array(A, dtype=float)
    orgA = np.copy(A)
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            for k in range(len(A[i])):
                if A[i, k] != 0:
                    factor = A[j, k] / A[i, k]
                    A[j] = factor * A[i] - A[j]
                    break
    cols_to_keep = []
    for row in A:
        for j in range(len(row)):
            if row[j] != 0:
                cols_to_keep.append(j)
                break
    return orgA[:, cols_to_keep]


def test_matrix_image():
    tests = [
        ("Basic rank deficient", 
         [[1, 2, 3], [2, 4, 6], [1, 3, 5]], 2),
        ("Leading zeros", 
         [[0, 1], [1, 0]], 2),
        ("Duplicate rows", 
         [[1, 0, 0], [1, 0, 0], [1, 0, 0]], 1),
        ("Zero row middle", 
         [[1, 2, 3], [0, 0, 0], [2, 4, 6]], 1),
        ("Wide matrix", 
         [[1, 0, 1, 0], [0, 1, 0, 1]], 2),
        ("Tall matrix", 
         [[1, 1], [1, 1], [2, 2], [0, 1]], 2),
        ("All zeros", 
         [[0, 0], [0, 0]], 0),
        ("Single nonzero", 
         [[5]], 1),
        ("Single zero", 
         [[0]], 0),
        ("Pivot ordering", 
         [[0, 0, 1], [0, 1, 0], [0, 1, 1]], 2),
        ("Three identical", 
         [[0, 1, 1], [0, 1, 1], [1, 0, 0]], 2),
        ("Column dependencies", 
         [[1, 2], [1, 3], [1, 4]], 2),
        ("Scattered pivots", 
         [[0, 0, 1], [1, 0, 0], [0, 1, 0]], 3),
        ("Rank 1 wide", 
         [[1, 2, 3, 4]], 1),
    ]

    passed = 0
    failed = 0

    for name, matrix, expected_rank in tests:
        A = np.array(matrix, dtype=float)
        result = matrix_image(A)
        actual_rank = result.shape[1] if result.size > 0 else 0

        if actual_rank == expected_rank:
            print(f"✓ {name}")
            passed += 1
        else:
            print(f"✗ {name}: expected rank {expected_rank}, got {actual_rank}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0

test_matrix_image()
