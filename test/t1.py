import torch
import numpy as np
from math import copysign, hypot
def upper_householder(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    R = np.copy(A)

    for cnt in range(1,num_rows - 1):
        x = R[cnt:, cnt-1]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -A[cnt, cnt])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_cnt = np.identity(num_rows)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

        A = np.dot(Q_cnt, A ).dot(Q_cnt)


    return A
def givens_rotation(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)#diag
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            r=hypot(R[col, col], R[row, col])
            c = R[col, col] / r
            s = -R[row, col] / r
            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)



A = np.array([[5, -3, 2],
              [6, -4, 4],
              [4, -4, 5]])
# Compute QR decomposition using Givens rotation
A=upper_householder(A)
print(A)
(Q, R) =givens_rotation(A)

# Print orthogonal matrix Q
print(Q)

# Print upper triangular matrix R
print(R)