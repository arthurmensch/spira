# Author: Mathieu Blondel, Arthur Mensch
# License: BSD
import datetime
import json
import os
import time
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
from joblib import load, Parallel, delayed
from matplotlib import gridspec

from sklearn import clone
from spira.completion import ExplicitMF, DictMF
from spira.cross_validation import train_test_split, ShuffleSplit, \
    cross_val_score
from spira.datasets import load_movielens
from spira.impl.dict_fact import csr_center_data

import seaborn.apionly as sns

def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    def __init__(self, X_tr, X_te, refit=False):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0
        self.refit = refit

    def __call__(self, mf):
        test_time = time.clock()
        # print(mf.Q_[1])
        if self.refit:
            if mf.normalize:
                if not hasattr(self, 'X_tr_c_'):
                    self.X_tr_c_, _, _ = csr_center_data(self.X_tr)
                else:
                    mf._refit_code(self.X_tr_c_)
            else:
                mf._refit_code(self.X_tr)
        X_pred = mf.predict(self.X_tr)
        loss = sqnorm(X_pred.data - self.X_tr.data) / 2
        regul = mf.alpha * (sqnorm(mf.P_))  # + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        self.rmse.append(rmse)

        # print(mf.B_[0])
        # print(mf.counter_)
        # self.q_values.append(mf.Q_[3, 0:100].copy())

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


def main(version='100k', n_jobs=1, random_state=0, cross_val=False):
    dl_params = {}
    dl_params['100k'] = dict(learning_rate=1, batch_size=10, offset=0, alpha=1)
    dl_params['1m'] = dict(learning_rate=.75, batch_size=100, offset=0,
                        alpha=.8)
    dl_params['10m'] = dict(learning_rate=.75, batch_size=500, offset=0,
                         alpha=3)
    dl_params['netflix'] = dict(learning_rate=.75, batch_size=4000, offset=0,
                         alpha=1.2)
    cd_params = {'100k': dict(alpha=.1), '1m': dict(alpha=.03), '10m': dict(alpha=.04),
                 'netflix': dict(alpha=.1)}

    if version in ['100k', '1m', '10m']:
        X = load_movielens(version)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
        X_tr = X_tr.tocsr()
        X_te = X_te.tocsr()
    elif version is 'netflix':
        X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
        X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))

    cd_mf = ExplicitMF(n_components=30, max_iter=50, alpha=.1, normalize=True,
                       verbose=1, )
    dl_mf = DictMF(n_components=30, n_epochs=5, alpha=1.17, verbose=5,
                   batch_size=10000, normalize=True,
                   fit_intercept=True,
                   random_state=0,
                   learning_rate=.75,
                   impute=False,
                   partial=False,
                   backend='c')
    dl_mf_partial = DictMF(n_components=30, n_epochs=5, alpha=1.17, verbose=5,
                   batch_size=10000, normalize=True,
                   fit_intercept=True,
                   random_state=0,
                   learning_rate=.75,
                   impute=False,
                   partial=True,
                   backend='c')

    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
    #                                              '-%M-%S')
    if cross_val:
        subdir = 'benches_ncv'
    else:
        subdir = 'benches'
    output_dir = expanduser(join('~/output/recommender/', subdir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alphas = np.logspace(-2, 1, 30)
    mf_list = [dl_mf_partial, dl_mf, cd_mf]
    dict_id = {cd_mf: 'cd', dl_mf: 'dl', dl_mf_partial: 'dl_partial'}
    names = {'cd': 'Coordinate descent', 'dl': 'Proposed online masked MF',
             'dl_partial': 'Proposed algorithm (with partial projection)'}

    if os.path.exists(join(output_dir, 'results_%s_%s.json' % (version,
                           random_state))):
        with open(join(output_dir, 'results_%s_%s.json' % (version,
                       random_state)), 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for mf in mf_list:
        results[dict_id[mf]] = {}
        if not cross_val:
            if isinstance(mf, DictMF):
                mf.set_params(learning_rate=dl_params[version]['learning_rate'],
                              batch_size=dl_params[version]['batch_size'],
                              alpha=dl_params[version]['alpha'])
            else:
                mf.set_params(alpha=cd_params[version]['alpha'])
        else:
            if isinstance(mf, DictMF):
                mf.set_params(learning_rate=dl_params[version]['learning_rate'],
                              batch_size=dl_params[version]['batch_size'])
            cv = ShuffleSplit(n_iter=3, train_size=0.66, random_state=0)
            mf_scores = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(single_fit)(mf, alpha, X_tr, cv) for alpha in alphas)
            mf_scores = np.array(mf_scores).mean(axis=1)
            best_alpha_arg = mf_scores.argmin()
            best_alpha = alphas[best_alpha_arg]
            mf.set_params(alpha=best_alpha)

        cb = Callback(X_tr, X_te, refit=isinstance(mf, DictMF))
        mf.set_params(callback=cb)
        mf.fit(X_tr)
        results[dict_id[mf]] = dict(name=names[dict_id[mf]],
                                    time=cb.times,
                                    rmse=cb.rmse)
        if cross_val:
            results['alphas'] = alphas.tolist()
            results['cv_alpha'] = mf_scores.tolist()
            results['best_alpha'] = mf.alpha

        with open(join(output_dir, 'results_%s_%s.json' % (version,
                                                           random_state)),
                  'w+') as f:
            json.dump(results, f)

        print('Done')


def single_fit(mf, alpha, X_tr, cv):
    mf_cv = clone(mf)
    if isinstance(mf_cv, DictMF):
        mf_cv.set_params(n_epochs=2)
    else:
        mf_cv.set_params(max_iter=10)
    mf_cv.set_params(alpha=alpha)
    score = cross_val_score(mf_cv, X_tr, cv)
    return score


def plot_benchs(output_dir=expanduser('~/output/recommender/benches')):
    fig = plt.figure()

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    ylims = {'100k': [.90, .96], '1m': [.865, .915], '10m': [.80, .868]}
    xlims = {'100k': [0.0001, 10], '1m': [0.05, 10], '10m': [0.5, 120]}

    for i, version in enumerate(['1m', '10m', 'netflix']):
        with open(join(output_dir, 'results_%s.json' % version), 'r') as f:
            data = json.load(f)
        ax_time = fig.add_subplot(gs[0, i])
        ax_time.grid()
        sns.despine(fig, ax_time)
        if i == 0:
            ax_time.set_ylabel('RMSE on test set')
        else:
            ax_time = fig.add_subplot(gs[0, i])
            # ax_time.set_xticklabels([])
            # plt.setp(ax_time.get_yticklabels(), visible=False)
        ax_time.grid()
        for estimator in data:
            this_data = data[estimator]
            ax_time.plot(this_data['time'], this_data['rmse'],
                         label=this_data['name'])
            # ax_tuning.plot(this_data['alpha'], this_data['cv_alpha'],
            #                label=this_data['name'])
        # ref_ax.legend()
        # ax_tuning.set_xscale('log')
        # ax_tuning.set_xlabel('$\\alpha$')
        # ax_tuning.set_ylabel('RMSE on test set')
        ax_time.set_xscale('log')
        ax_time.set_xlabel('CPU time')
        ax_time.set_ylim(ylims[version])
        ax_time.set_xlim(xlims[version])
        ax_time.set_title('Movielens %s' % version)
        # ax_time.set_xlim([1e-2, 1e2])

        # ax_tuning.legend()
    plt.tight_layout()
    plt.savefig('output.pdf')


if __name__ == '__main__':
    main('netflix', n_jobs=30, cross_val=True)
    # main('100k', n_jobs=3, cross_val=True)
    # for i in range(5):
    #     main('1m', cross_val=True, n_jobs=15, random_state=i)
    #     main('10m', n_jobs=15, random_state=i, cross_val=True)
    # plot_benchs()
