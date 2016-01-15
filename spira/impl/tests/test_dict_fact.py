import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal

from spira.completion import DictMF

from testing import assert_array_almost_equal
from testing import assert_almost_equal

# from spira.impl.dict_fact import _sample


def test_matrix_fact_cd():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = DictMF(n_components=3, n_epochs=3, alpha=1e-3, random_state=0,
                verbose=0, normalize=False)

    mf.fit(X)

    Y = np.dot(mf.P_, mf.Q_)
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)

def test_dict_fact_normalize():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    mf = DictMF(n_components=3, n_epochs=1, alpha=1e-3, random_state=0,
                verbose=0, normalize=True)

    mf.fit(X)

    Y = np.dot(mf.P_, mf.Q_)
    Y += mf.col_mean_[np.newaxis, :]
    Y += mf.row_mean_[:, np.newaxis]
    Y2 = mf.predict(X).toarray()

    assert_array_almost_equal(Y, Y2)

    rmse = np.sqrt(np.mean((X - Y) ** 2))
    rmse2 = mf.score(X)

    assert_almost_equal(rmse, rmse2)


# def test_sample():
#     data = np.ones(5)
#     row = np.arange(5)
#     col = np.arange(0, 10, 2)
#     X = sp.coo_matrix((data, (row, col)), shape=(5, 10))
#     X = sp.csr_matrix(X)
#     y, idx, count = _sample(X.data, X.indices, X.indptr, 10, np.array([0, 1]))
#     assert_array_equal(y, np.array([[1., 0.], [0., 1.]]))
#     assert_array_equal(idx, np.array([0, 2]))
#     assert_array_equal(count, np.array([1, 1]))
