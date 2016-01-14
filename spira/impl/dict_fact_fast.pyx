import cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt
from scipy import linalg

cdef extern from "../src/cblas/cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                double *X, int incX, double *Y, int incY, double *A, int lda) nogil

    void dgemm "cblas_dgemm"(CBLAS_ORDER Order,CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, double *A,
                 int lda, double *B, int ldb,
                 double beta, double *C, int ldc);


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void _update_dict_fast(double[:, :] Q, double[:, :] A, double[:, :] R,
                 bint fit_intercept, long[:] components_range,
                                     double[:] norm):

    cdef unsigned int n_components = Q.shape[0]
    cdef unsigned int n_cols = Q.shape[1]
    cdef unsigned int components_range_len = components_range.shape[0]
    # Q, A, B c-ordered
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double new_norm
    cdef unsigned int j

    for idx in range(components_range_len):
        j = components_range[idx]
        norm[j] = 0
        for k in range(n_cols):
            norm[j] += Q[j, k] * Q[j, k]
        norm[j] = sqrt(norm[j])

    for idx in range(components_range_len):
        j = components_range[idx]
        dger(CblasRowMajor, n_components, n_cols, 1.0,
             A_ptr + j * n_components,
             1, Q_ptr + j * n_cols, 1, R_ptr, n_cols)
        new_norm = 0
        for k in range(n_cols):
            Q[j, k] = R[j, k] / A[j, j]
            new_norm += Q[j, k] ** 2
        new_norm = sqrt(new_norm) / norm[j]
        if new_norm > 1:
            for k in range(n_cols):
                Q[j, k] /= new_norm
        dger(CblasRowMajor, n_components, n_cols, -1.0,
             A_ptr + j * n_components,
             1, Q_ptr + j * n_cols, 1, R_ptr, n_cols)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _online_dl_fast(double[:] X_data, int[:] X_indices, int[:] X_indptr,
               int n_cols,
               double alpha, double learning_rate,
               double[:, : ] A, double[:, : ] B, int seen_rows, int[:] seen_cols, int n_iter,
               double[:, : ] P, double[:, : ] Q,
               bint fit_intercept, int n_epochs, int random_seed, bint verbose,
               object callback):
    cdef int n_rows = len(X_indptr) - 1
    cdef int n_components = Q.shape[0]
    cdef int i, j, k
    cdef int[:] idx
    cdef double[:] y
    cdef int col_nnz
    cdef double w_A
    cdef double[:] w_B
    cdef double[:, :] R
    cdef long[:] components_range
    cdef int[:] row_range
    cdef double[:] norm

    cdef double[:, :] Q_idx
    cdef double[:, :] R_idx
    cdef int[:] seen_cols_idx

    cdef double* A_ptr = &A[0, 0]
    cdef double* Q_idx_ptr
    cdef double* R_idx_ptr

    np.random.seed(random_seed)
    row_range = np.arange(n_rows, dtype=np.int32)
    norm = np.zeros(n_components)

    if not fit_intercept:
        components_range = np.empty(n_components, dtype=np.int64)
    else:
        components_range = np.empty(n_components - 1, dtype=np.int64)
    for _ in range(n_epochs):
        np.random.shuffle(row_range)
        for j in row_range:
            idx = X_indices[X_indptr[j]:X_indptr[j + 1]]
            y = X_data[X_indptr[j]:X_indptr[j + 1]]
            col_nnz = len(idx)

            if col_nnz != 0:
                Q_idx = np.empty((n_components, col_nnz), order='c',
                                 dtype=np.float64)
                Q_idx_ptr = &Q[0, 0]

                for i, k in enumerate(idx):
                    Q_idx[:, i] = Q[:, k]

                # P[j] = _solve_cholesky(
                #         np.array(Q_idx).T,
                #                         np.array(y),
                #                         2 * alpha
                #                         * col_nnz / n_cols).view(dtype=np.float64)

                seen_rows += 1

                seen_cols_idx = np.empty(col_nnz, dtype=np.int32)
                for i, k in enumerate(idx):
                    seen_cols[k] += 1
                    seen_cols_idx[j] = seen_cols[k]

                w_A = pow(seen_rows, -learning_rate)
                w_B = np.power(seen_cols_idx,
                               -learning_rate)

                for i in range(n_components):
                    for k in range(n_components):
                        A[i, k] *= 1 - w_A
                        A[i, k] += P[j, i] * P[j, k] * w_A

                for i in range(n_components):
                    for k in idx:
                        B[j, k] *= 1 - w_B[j]
                        A[i, k] += P[j, i] * y[k] * w_B[j]

                R_idx = np.empty((n_components, col_nnz), order='c', dtype=np.float64)
                R_idx_ptr = &R_idx[0, 0]

                for i, k in enumerate(idx):
                    R_idx[:, i] = B[:, k]

                # R_idx[:] = np.array(R_idx) - np.dot(np.array(A), np.array(Q_idx))
                dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      n_components, col_nnz, n_components,
                      -1,
                      A_ptr, n_components, Q_idx_ptr, col_nnz,
                      1,
                      R_idx_ptr, col_nnz)

                if not fit_intercept:
                    components_range = np.random.permutation(n_components)
                else:
                    components_range = np.random.permutation(n_components - 1)
                    for i in range(n_components - 1):
                        components_range[i] += 1

                _update_dict_fast(
                        Q_idx,
                        A,
                        R_idx,
                        fit_intercept,
                        components_range,
                        norm)

                for i in idx:
                    Q[:, i] = Q_idx[:, i]


            if verbose:
                if n_iter % 5000 == 0:
                    print("Iteration %i" % n_iter)
                    callback()
            n_iter += 1
