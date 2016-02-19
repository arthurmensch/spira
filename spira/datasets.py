# Author: Mathieu Blondel
# License: BSD
import os

from sklearn.datasets.base import get_data_home as _get_data_home

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


def get_data_home():
    return _get_data_home().replace("scikit_learn", "spira")


def load_movielens(version):
    data_home = get_data_home()

    if version == "100k":
        path = os.path.join(data_home, "movielens100k", "movielens100k.pkl")
    elif version == "1m":
        path = os.path.join(data_home, "movielens1m", "movielens1m.pkl")
    elif version == "10m":
        path = os.path.join(data_home, "movielens10m", "movielens10m.pkl")
    else:
        raise ValueError("Invalid version of movielens.")

    # FIXME: make downloader
    if not os.path.exists(path):
        raise ValueError("Dowload dataset using 'make download-movielens%s' at"
                         " project root." % version)

    X = joblib.load(path)
    return X

def load_netflix():
    path = os.path.join(data_home, "netflix", "netflix.pkl")
    X = joblib.load(path)
    probe = os.path.join(data_home, "netflix", "netflix.pkl")
    X = joblib.load(path)
    return X