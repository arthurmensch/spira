# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport sqrt

from scipy.linalg.cython_blas cimport dger, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dposv

from libc.math cimport pow

import numpy as np

cdef char UP = 'U'
cdef char NTRANS = 'N'
cdef char TRANS = 'T'
cdef int zero = 0
cdef int one = 1
cdef double zerod = 0
cdef double oned = 1
cdef double moned = -1

ctypedef np.uint32_t UINT32_t

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef void _shuffle(long[:] arr, UINT32_t* random_state):
    cdef int len_arr = arr.shape[0]
    cdef int i, j
    for i in range(len_arr -1, 0, -1):
        j = rand_int(i + 1, random_state)
        arr[i], arr[j] = arr[j], arr[i]


cpdef int _update_code_full_fast(double[:] X_data, int[:] X_indices,
                  int[:] X_indptr, long n_rows, long n_cols,
                  double alpha, double[::1, :] P, double[::1, :] Q,
                  double[::1, :] Q_idx,
                  double[::1, :] C):
    cdef int n_components = P.shape[0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* G_ptr = &C[0, 0]
    cdef double* X_data_ptr = &X_data[0]
    cdef int info = 0
    cdef int ii, jj
    cdef int nnz
    cdef double reg

    for ii in range(n_rows):
        nnz = X_indptr[ii + 1] - X_indptr[ii]
        # print('Filling Q')

        for jj in range(nnz):
            Q_idx[:, jj] = Q[:, X_indices[X_indptr[ii] + jj]]
        # print('Computing Gram')

        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &nnz,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &zerod,
              G_ptr, &n_components
              )
        # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        reg = 2 * alpha * nnz / n_cols
        for p in range(n_components):
            C[p, p] += reg

        # print('Computing Q**T x')
        # Qx = Q_idx.dot(x)
        dgemv(&NTRANS,
              &n_components, &nnz,
              &oned,
              Q_idx_ptr, &n_components,
              X_data_ptr + X_indptr[ii], &one,
              &zerod,
              P_ptr + ii * n_components, &one
              )

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        # print('Solving linear system')
        dposv(&UP, &n_components, &one, G_ptr, &n_components,
              P_ptr + ii * n_components, &n_components,
              &info)
        if info != 0:
            return -1
    return 0


cdef int _update_code_fast(double[:] X_data, int[:] X_indices,
                  int[:] X_indptr, long n_rows, long n_cols,
                  double alpha, double learning_rate,
                  double[::1, :] A, double[::1, :] B,
                  double[::1, :] G, double[::1, :] T,
                  long[:] counter,
                  double[::1, :] P, double[::1, :] Q,
                  long[:] row_batch,
                  double[::1, :] C,
                  double[::1, :] Q_idx,
                  double[::1, :] P_batch,
                  double[:] sub_Qx,
                  char[:] idx_mask,
                  long[:] idx_concat,
                  bint impute):
    cdef int len_batch = row_batch.shape[0], n_components = P.shape[0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* P_batch_ptr = &P_batch[0, 0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* P_ptr = &P[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* C_ptr = &C[0, 0]
    cdef double* X_data_ptr = &X_data[0]
    cdef int info = 0
    cdef int ii, jj, i, j, k, m
    cdef int nnz
    cdef double reg, v
    cdef int last = 0
    cdef double w_B, w_A, mu_A

    for jj in range(n_cols):
        idx_mask[jj] = 0

    for ii in range(len_batch):
        i = row_batch[ii]
        nnz = X_indptr[i + 1] - X_indptr[i]
        # print('Filling Q')

        for jj in range(nnz):
            Q_idx[:, jj] = Q[:, X_indices[X_indptr[i] + jj]]
        # print('Computing Gram')

        if impute:
            reg = 2 * alpha
            v = 1 # nnz / n_cols
            for p in range(n_components):
                sub_Qx[p] = 0
                for jj in range(nnz):
                    j = X_indices[X_indptr[i] + jj]
                    T[p, 0] -= T[p, j + 1]
                    T[p, j + 1] = Q_idx[p, jj] * X_data[X_indptr[i] + jj]
                    sub_Qx[p] += T[p, j + 1]
                T[p, 0] += sub_Qx[p]
                P_batch[p, ii] = (1 - v) * sub_Qx[p] + v * T[p, 0]
            for p in range(n_components):
                for n in range(n_components):
                    C[p, n] = G[p, n]
        else:
            dgemm(&NTRANS, &TRANS,
                  &n_components, &n_components, &nnz,
                  &oned,
                  Q_idx_ptr, &n_components,
                  Q_idx_ptr, &n_components,
                  &zerod,
                  C_ptr, &n_components
                  )
            reg = 2 * alpha * nnz / n_cols

            # print('Computing Q**T x')
            # Qx = Q_idx.dot(x)
            dgemv(&NTRANS,
                  &n_components, &nnz,
                  &oned,
                  Q_idx_ptr, &n_components,
                  X_data_ptr + X_indptr[i], &one,
                  &zerod,
                  P_batch_ptr + ii * n_components, &one
                  )

        # C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        for p in range(n_components):
            C[p, p] += reg

        # P[j] = linalg.solve(C, Qx, sym_pos=True,
        #                     overwrite_a=True, check_finite=False)
        # print('Solving linear system')
        dposv(&UP, &n_components, &one, C_ptr, &n_components,
              P_batch_ptr + ii * n_components, &n_components,
              &info)
        if info != 0:
            raise ValueError

        # A *= 1 - w_A * len_batch
        # A += P[row_batch].T.dot(P[row_batch]) * w_A
        counter[0] += 1
        w_A = pow(counter[0], -learning_rate)
        mu_A = 1 - w_A
        for k in range(n_components):
            for m in range(n_components):
                A[k, m] *= mu_A
        dger(&n_components, &n_components,
             &w_A,
             P_batch_ptr + ii * n_components, &one,
             P_batch_ptr + ii * n_components, &one,
             A_ptr, &n_components)

        # w_B = np.power(counter[1][idx], - learning_rate)[np.newaxis, :]
        # B[:, idx] *= 1 - w_B
        # B[:, idx] += np.outer(P[j], x) * w_B
        # Use a loop to avoid copying a contiguous version of B
        for jj in range(nnz):
            j = X_indices[X_indptr[i] + jj]
            idx_mask[j] = 1
            # counter[j + 1] += 1
            # w_B = pow(counter[j + 1], -learning_rate)
            for k in range(n_components):
                B[k, j] = (1 - w_A) * B[k, j] + \
                                 w_A * P_batch[k, ii]\
                                 * X_data[X_indptr[i] + jj]
        # dger(&n_components, &nnz,
        #      &w_B,
        #      P_batch_ptr + ii * n_components, &one,
        #      X_data + X_indptr[i], &one,
        #      &mu_B,
        #      &B_batch)

        P[:, i] = P_batch[:, ii]

    for ii in range(n_cols):
        if idx_mask[ii]:
            idx_concat[last] = ii
            last += 1

    return last


cdef void _update_dict_fast(double[::1, :] A, double[::1, :] B,
                             double[::1, :] G,
                             double[::1, :] Q,
                             double[::1, :] R,
                             double[::1, :] Q_idx,
                             double[::1, :] old_sub_G,
                             long[:] idx,
                             bint fit_intercept, long[:] components_range,
                             double[:] norm,
                             bint impute):

    cdef int n_components = Q.shape[0]
    cdef int idx_len = idx.shape[0]
    cdef unsigned int components_range_len = components_range.shape[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* Q_idx_ptr = &Q_idx[0, 0]
    cdef double* A_ptr = &A[0, 0]
    cdef double* R_ptr = &R[0, 0]
    cdef double* G_ptr = &G[0, 0]
    cdef double* old_sub_G_ptr = &old_sub_G[0, 0]
    cdef double new_norm
    cdef unsigned int k, kk, j, jj

    for jj in range(idx_len):
        j = idx[jj]
        R[:, jj] = B[:, j]
        Q_idx[:, jj] = Q[:, j]

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &idx_len,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &zerod,
              old_sub_G_ptr, &n_components
              )

    dgemm(&NTRANS, &NTRANS,
          &n_components, &idx_len, &n_components,
          &moned,
          A_ptr, &n_components,
          Q_idx_ptr, &n_components,
          &oned,
          R_ptr, &n_components)

    for kk in range(components_range_len):
        k = components_range[kk]
        norm[k] = 0
        for jj in range(idx_len):
            norm[k] += Q_idx[k, jj] * Q_idx[k, jj]
        norm[k] = sqrt(norm[k])

    for kk in range(components_range_len):
        k = components_range[kk]
        dger(&n_components, &idx_len, &oned,
             A_ptr + k * n_components,
             &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)
        new_norm = 0
        for jj in range(idx_len):
            Q_idx[k, jj] = R[k, jj] / A[k, k]
            new_norm += Q_idx[k, jj] ** 2
        new_norm = sqrt(new_norm) / norm[k]
        if new_norm > 1:
            for jj in range(idx_len):
                Q_idx[k, jj] /= new_norm
        # R -= A[:, k] Q[:, k].T
        dger(&n_components, &idx_len, &moned,
             A_ptr + k  * n_components,
             &one, Q_idx_ptr + k, &n_components, R_ptr, &n_components)
    for jj in range(idx_len):
        j = idx[jj]
        Q[:, j] = Q_idx[:, jj]

    if impute:
        dgemm(&NTRANS, &TRANS,
              &n_components, &n_components, &idx_len,
              &oned,
              Q_idx_ptr, &n_components,
              Q_idx_ptr, &n_components,
              &oned,
              G_ptr, &n_components
              )
        for k in range(n_components):
            for j in range(n_components):
                G[j, k] -= old_sub_G[j, k]


def _online_dl_fast(double[:] X_data, int[:] X_indices,
                    int[:] X_indptr, long n_rows, long n_cols,
                    long[:] row_range,
                    long max_idx_size,
                    double alpha, double learning_rate,
                    double[::1, :] A, double[::1, :] B,
                    long[:] counter,
                    double[::1, :] G, double[::1, :] T,
                    double[::1, :] P, double[::1, :] Q,
                    long n_epochs, long batch_size,
                    UINT32_t random_seed,
                    long verbose,
                    bint fit_intercept,
                    bint impute,
                    callback):


    cdef int n_batches = n_rows // batch_size
    cdef int n_cols_int = n_cols
    cdef int n_components = P.shape[0]
    cdef UINT32_t seed = random_seed
    cdef double[::1, :] Q_idx = np.zeros((n_components, max_idx_size),
                                         order='F')
    cdef double[::1, :] R = np.zeros((n_components, max_idx_size),
                                     order='F')
    cdef double[::1, :] P_batch = np.zeros((n_components, batch_size),
                                           order='F')
    cdef double[::1, :] C = np.zeros((n_components, n_components), order='F')
    cdef double[:] sub_Qx = np.zeros(n_components)
    cdef double[::1, :] old_sub_G = np.zeros((n_components, n_components),
                                             order='F')
    cdef char[:] idx_mask = np.zeros(n_cols, dtype='i1')
    cdef long[:] idx_concat = np.zeros(max_idx_size, dtype='int')
    cdef long[:] components_range
    cdef double[:] norm = np.zeros(n_components)
    cdef int i, start, stop, last, last_call = 0
    cdef long[:] row_batch

    cdef double* Q_ptr = &Q[0, 0]
    cdef double* G_ptr = &G[0, 0]

    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)

    for _ in range(n_epochs):
        _shuffle(row_range, &seed)
        for i in range(n_batches):
            start = i * batch_size
            stop = start + batch_size
            if stop > n_rows:
                stop = n_rows
            row_batch = row_range[start:stop]
            last = _update_code_fast(X_data, X_indices,
                                     X_indptr, n_rows, n_cols,
                                     alpha, learning_rate,
                                     A, B, G, T,
                                     counter,
                                     P, Q,
                                     row_batch,
                                     C,
                                     Q_idx,
                                     P_batch,
                                     sub_Qx,
                                     idx_mask,
                                     idx_concat,
                                     impute)
            _shuffle(components_range, &seed)
            _update_dict_fast(
                    A,
                    B,
                    G,
                    Q,
                    R,
                    Q_idx,
                    old_sub_G,
                    idx_concat[:last],
                    fit_intercept,
                    components_range,
                    norm,
                    impute)
            if verbose and counter[0] // (n_rows // verbose) == last_call + 1:
                    print("Iteration %i" % (counter[0]))
                    last_call += 1
                    callback(G)