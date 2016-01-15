import cython
import numpy as np

from libc.math cimport sqrt

from scipy.linalg.cython_blas cimport dger, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dposv

from libc.math cimport pow


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _update_code(double[:] X_data, long[:] X_indices,
                  long[:] X_indptr, int n_rows, int n_cols,
                  double alpha, double learning_rate,
                  double[::1, :] A, double[::1, :] B,
                  long[:] counter,
                  double[::1, :] P, double[::1, :] Q,
                  int[:] row_batch,
                  double[::1, :] Q_idx,
                  double[::1, :] P_batch,
                  double[::1, 1] C,
                  bint[:] idx_mask,
                  int[:] idx_concat):
    cdef int len_batch = row_batch.shape[0], n_components = P.shape[1]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* P_batch_ptr = &P_batch[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* C_ptr = &C[0, 0]
    cdef double* X_data_ptr = &X_data[0, 0]
    cdef char up = 'U'
    cdef bint true = True
    cdef bint false = False
    cdef int zero = 0
    cdef int one = 1
    cdef int info = 0
    cdef double zerod = 0
    cdef double oned = 1
    cdef int ii, jj, k, idx_in_B

    for jj in range(n_cols):
        idx_mask[jj] = 0

    for ii in range(len_batch):
        i = row_batch[ii]
        nnz = X_indptr[i + 1] - X_indptr[i]
        idx = X_indices[X_indptr[j]:X_indptr[i + 1]]

        Q_idx[:, len_batch] = Q[:, idx]

        # C = Q_idx.dot(Q_idx.T)
        dgemm(&False, &True,
              &n_components, &n_components, &nnz,
              &oned,
              Q_idx_ptr, &one, &n_components,
              Q_idx_ptr, &one, &n_components,
              &zerod,
              C_ptr, &n_components
              )

        # Qx = Q_idx.dot(x)
        dgemv(&False,
              &n_components, &nnz,
              &oned,
              Q_idx_ptr, &1, &n_components,
              X_data_ptr + X_indptr[i], &1,
              &zerod,
              P_batch_ptr + ii * n_components, &1
              )

        # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        reg = 2 * alpha * nnz / n_cols
        for p in range(n_components):
            C[p, p] += reg

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        dposv(&up, &n_components, &1, C_ptr, &n_components,
              P_batch_ptr + ii * n_components, &n_components,
              &info)
        if info != 0:
            return -1

        # w_B = np.power(counter[1][idx], - learning_rate)[np.newaxis, :]
        # B[:, idx] *= 1 - w_B
        # B[:, idx] += np.outer(P[j], x) * w_B
        for jj in range(nnz):
            j = idx[jj]
            idx_mask[j] = 1
            counter[j + 1] += 1
            w_B = pow(counter[idx_in_B + 1], -learning_rate)
            for k in range(n_components):
                B[k, j] = (1 - w_B) * B[k, idx] + \
                                 w_B * P_batch[ii, k]\
                                 * X_data[X_indptr[i] + jj]

    last = 0
    for i in range(n_cols):
        if idx_mask[i]:
            idx_concat[last] = i
            last += 1

    counter[0] += len_batch
    w_A = pow(counter[0], -learning_rate)
    mu_A = 1 - w_A * len_batch

    # A *= 1 - w_A * len_batch
    # A += P[row_batch].T.dot(P[row_batch]) * w_A
    dgemm(&False, &True,
          &n_components, &n_components, &len_batch,
          &w_A,
          P_batch_ptr, &1, &n_components,
          P_batch_ptr, &1, &n_components,
          &mu_A,
          A_ptr, &n_components)

    return last

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void _update_dict_fast(double[::1, :] Q, double[::1, :] A,
                             double[::1, :] R,
                             bint fit_intercept, long[:] components_range,
                                     double[:] norm):

    cdef int n_components = Q.shape[0]
    cdef int n_cols = Q.shape[1]
    cdef unsigned int components_range_len = components_range.shape[0]
    # Q, A, B c-ordered
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double new_norm
    cdef unsigned int j
    cdef int incA = 1
    cdef double alpha = 1.0, malpha = -1.

    for idx in range(components_range_len):
        j = components_range[idx]
        norm[j] = 0
        for k in range(n_cols):
            norm[j] += Q[j, k] * Q[j, k]
        norm[j] = sqrt(norm[j])

    for idx in range(components_range_len):
        j = components_range[idx]
        # R += A[:, j] Q[:, j].T
        dger(&n_components, &n_cols, &alpha,
             A_ptr + j * n_components,
             &incA, Q_ptr + j, &n_components, R_ptr, &n_components)
        new_norm = 0
        for k in range(n_cols):
            Q[j, k] = R[j, k] / A[j, j]
            new_norm += Q[j, k] ** 2
        new_norm = sqrt(new_norm) / norm[j]
        if new_norm > 1:
            for k in range(n_cols):
                Q[j, k] /= new_norm
        # R -= A[:, j] Q[:, j].T
        dger(&n_components, &n_cols, &malpha,
             A_ptr + j  * n_components,
             &incA, Q_ptr + j, &n_components, R_ptr, &n_components)
