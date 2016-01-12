from math import sqrt

import numpy as np
import scipy.sparse as sp
from numba import jit
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.linear_model import ridge_regression
from sklearn.utils import check_random_state
from sklearn_recommender.base import csr_center_data
from spira.impl.matrix_fact_fast import _predict
from spira.metrics import rmse


class DictMF(BaseEstimator):
    def __init__(self, alpha=1.0, learning_rate=1,
                 n_components=30, n_epochs=2,
                 normalize=False,
                 fit_intercept=False,
                 callback=None, random_state=None, verbose=0):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.callback = callback
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def _init(self, X, random_state):
        n_rows, n_cols = X.shape

        # P code
        P = np.zeros((n_rows, self.n_components), order='C')

        # Q dictionary
        Q = np.empty((self.n_components, n_cols), order='F')
        if self.fit_intercept:
            # Intercept on first col
            Q[0] = 1
            Q[1:] = random_state.randn(self.n_components - 1, n_cols)
        else:
            Q[:] = random_state.randn(self.n_components, n_cols)

        S = np.sqrt(np.sum(Q ** 2, axis=1))
        Q /= S[:, np.newaxis]

        return P, Q

    def _refit_code(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        _, n_cols = X.shape
        _online_refit(X.data, X.indices, X.indptr,
                      n_cols,
                      self.alpha, self.P_, self.Q_)

    def fit(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        n_rows, n_cols = X.shape

        random_state = check_random_state(self.random_state)

        self.P_, self.Q_ = self._init(X, random_state)

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='C')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.seen_rows_ = 0
        self.seen_cols_ = np.zeros(n_cols)
        self.n_iter_ = 0

        if self.normalize:
            (X, self.global_mean_,
             self.row_mean_, self.col_mean_) = csr_center_data(X)

        row_range = np.arange(n_rows)
        for k in range(self.n_epochs):
            random_state.shuffle(row_range)
            for j in row_range:

                idx_range = slice(X.indptr[j], X.indptr[j + 1])
                idx = X.indices[idx_range]
                y = X.data[idx_range]
                col_nnz = len(idx)

                if col_nnz != 0:
                    self.P_[j] = ridge_regression(self.Q_[:, idx].T,
                                                 y,
                                                 2 * self.alpha
                                                 * col_nnz / n_cols)

                    self.seen_rows_ += 1
                    self.seen_cols_[idx] += 1

                    w_A = pow(self.seen_rows_, -self.learning_rate)
                    w_B = np.power(self.seen_cols_[idx],
                                   -self.learning_rate)[np.newaxis, :]

                    self.A_ *= 1 - w_A
                    self.A_ += np.outer(self.P_[j], self.P_[j]) * w_A
                    self.B_[:, idx] *= 1 - w_B
                    self.B_[:, idx] += np.outer(self.P_[j], y) * w_B

                    _update_dict(
                            self.Q_,
                            self.A_,
                            self.B_,
                            idx,
                            self.fit_intercept,
                            np.random.randint(0, np.iinfo(np.uint32).max))

                if self.verbose:
                    if self.n_iter_ % 100 == 0:
                        print("Iteration %i" % self.n_iter_)
                        # self._refit_code(X)
                        self._callback()
                self.n_iter_ += 1


                # random_seed = random_state.randint(0, np.iinfo(np.uint32).max)
                # self.P_, self.Q_ = _online_dl(X.data, X.indices, X.indptr,
                #                                 n_cols,
                #                                 self.alpha, self.A_, self.B_,
                #                                 self.seen_cols_, self.P_, self.Q_,
                #                                 self.n_epochs,
                #                                 random_seed, self.verbose,
                #                                 self._callback)
    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_, np.asfortranarray(self.Q_))

        if self.normalize:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices,
                                       mode='clip') + self.global_mean_

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


@jit("void(f8[:], u4[:], u4[:], u4,"
     "f8, f8[:, ::1], f8[::1, :])")
def _online_refit(X_data, X_indices, X_indptr, n_cols,
                  alpha, P, Q):
    n_rows = len(X_indptr) - 1
    for j in range(n_rows):
        idx_range = slice(X_indptr[j], X_indptr[j + 1])
        idx = X_indices[idx_range]
        y = X_data[idx_range]
        col_nnz = len(idx)
        if col_nnz != 0:
            P[j] = _solve_cholesky(Q[:, idx].T,
                                   y,
                                   alpha * col_nnz / n_cols)


@jit("pyobject(f8[:], u4[:], u4[:], u4,"
     "f8, f8[:, ::1], f8[::1, :],"
     "i8[:], f8[:, ::1], f8[::1, :], i8, u4, i1, pyobject)")
def _online_dl(X_data, X_indices, X_indptr, n_cols,
               alpha, A, B,
               seen_cols, P, Q, n_epochs, random_seed, verbose, callback):
    n_rows = len(X_indptr) - 1

    np.random.seed(random_seed)
    row_range = np.arange(n_rows)
    for k in n_epochs:
        np.random.shuffle(row_range)
        for n_iter, j in enumerate(row_range):
            n_iter += k * n_rows
            idx_range = slice(X_indptr[j], X_indptr[j + 1])
            idx = X_indices[idx_range]
            y = X_data[idx_range]
            col_nnz = len(idx)

            if col_nnz != 0:
                P[j] = _solve_cholesky(Q[:, idx].T,
                                       y,
                                       alpha * col_nnz / n_cols)

                seen_cols[-1] += 1
                seen_cols[idx] += 1
                A *= 1 - 1. / sqrt(seen_cols[-1])
                A += np.outer(P[j], P[j]) / sqrt(seen_cols[-1])
                B[:, idx] *= (1 - 1 / np.sqrt(seen_cols[idx]))[np.newaxis, :]
                B[:, idx] += np.outer(P[j], y) * (1 / np.sqrt(seen_cols[idx]))[
                                                 np.newaxis, :]
                Q[:, idx] = _update_dict(Q[:, idx],
                                         A,
                                         B[:, idx],
                                         np.random.randint(0, np.iinfo(
                                                 np.uint32).max))
            if verbose:
                if n_iter % 500 == 0:
                    print("Iteration %i" % n_iter)
                    callback()

    return A, B, seen_cols, P, Q


@jit("f8[:](f8[:, :], f8[:], f8)")
def _solve_cholesky(X, y, alpha):
    _, n_features = X.shape
    A = X.T.dot(X)
    Xy = X.T.dot(y)
    A.flat[::n_features + 1] += alpha
    return linalg.solve(A, Xy, sym_pos=True,
                        overwrite_a=True).T


@jit("f8[:, :](f8[::1, :], f8[:, ::1], u4[:], f8[::1, :], i1, u4)")
def _update_dict(Q, A, B, idx, fit_intercept, random_seed):
    n_components = Q.shape[0]
    np.random.seed(random_seed)

    norm = np.empty(n_components)
    for j in range(n_components):
        norm[j] = np.sqrt(np.sum(Q[j, idx] ** 2))
        if norm[j] == 0:
            norm[j] = 1

    # Intercept on first column
    if fit_intercept:
        components_range = np.arange(1, n_components)
    else:
        components_range = np.arange(n_components)
    np.random.shuffle(components_range)
    for j in components_range:
        Q[j, idx] += (B[j, idx] - np.dot(A[j], Q[:, idx])) / A[j, j]
        new_norm = np.sqrt(np.sum(Q[j, idx] ** 2))
        if new_norm > norm[j]:
            Q[j, idx] /= new_norm / norm[j]
