import numpy as np

from spira.preprocessing import StandardScaler

from testing import assert_array_equal
from testing import assert_array_almost_equal

X = [[3, 0, 0, 1],
     [2, 0, 5, 0],
     [0, 4, 3, 0],
     [0, 0, 2, 0],
     [1, 0, 0, 0]]

X = np.array(X, dtype=np.float64)


def _mean_axis1(X):
    mean = []
    for i in range(X.shape[0]):
        x = X[i]
        x = x[x != 0]
        mean.append(np.mean(x))
    return np.array(mean)


def _std_axis1(X):
    std = []
    for i in range(X.shape[0]):
        x = X[i]
        x = x[x != 0]
        std.append(np.std(x))
    std = np.array(std)
    std[std == 0] = 1
    return std


def _mean_axis0(X):
    mean = []
    for j in range(X.shape[1]):
        x = X[:, j]
        x = x[x != 0]
        mean.append(np.mean(x))
    return np.array(mean)


def _std_axis0(X):
    std = []
    for j in range(X.shape[1]):
        x = X[:, j]
        x = x[x != 0]
        std.append(np.std(x))
    std = np.array(std)
    std[std == 0] = 1
    return std


def test_center_std_axis1():
    X_t = X.copy()

    mean = _mean_axis1(X)
    std = _std_axis1(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == 0:
                continue
            X_t[i, j] -= mean[i]
            X_t[i, j] /= std[i]

    scaler = StandardScaler(with_std=True)
    X_t2 = scaler.fit_transform(X)

    assert_array_almost_equal(X_t, X_t2.toarray())

    X2 = scaler.inverse_transform(X_t2)
    assert_array_almost_equal(X, X2.toarray())


def test_center_std_axis0():
    X_t = X.copy()

    mean = _mean_axis0(X)
    std = _std_axis0(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == 0:
                continue
            X_t[i, j] -= mean[j]
            X_t[i, j] /= std[j]

    scaler = StandardScaler(axis=0, with_std=True)
    X_t2 = scaler.fit_transform(X)

    assert_array_almost_equal(X_t, X_t2.toarray())

    X2 = scaler.inverse_transform(X_t2)
    assert_array_almost_equal(X, X2.toarray())
