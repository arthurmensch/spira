# Author: Mathieu Blondel, Arthur Mensch
# License: BSD
from collections import OrderedDict

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
from spira.cross_validation import train_test_split, ShuffleSplit,\
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

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


def main(version='100k', n_jobs=1, random_state=0, cross_val=False):
    dl_params = {}
    dl_params['100k'] = dict(learning_rate=1, batch_size=10, offset=0, alpha=1)
    dl_params['1m'] = dict(learning_rate=.8, batch_size=50, offset=0,
                        alpha=.8)
    dl_params['10m'] = dict(learning_rate=.8, batch_size=250, offset=0,
                         alpha=3)
    dl_params['netflix'] = dict(learning_rate=1, batch_size=4000, offset=0,
                         alpha=0.2)
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
    dl_mf = DictMF(n_components=30, n_epochs=10, alpha=1.17, verbose=5,
                   batch_size=10000, normalize=True,
                   fit_intercept=True,
                   random_state=0,
                   learning_rate=.75,
                   impute=False,
                   partial=False,
                   backend='c')
    dl_mf_partial = DictMF(n_components=30, n_epochs=10, alpha=1.17, verbose=5,
                   batch_size=10000, normalize=True,
                   fit_intercept=True,
                   random_state=0,
                   learning_rate=.75,
                   impute=False,
                   partial=True,
                   backend='c')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    if cross_val:
        subdir = 'benches_ncv'
    else:
        subdir = 'benches'
    output_dir = expanduser(join('~/output/recommender/', timestamp,    subdir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alphas = np.logspace(-3, 0, 10)
    mf_list = [dl_mf_partial, dl_mf]
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
            if version != 'netflix':
                cv = ShuffleSplit(n_iter=3, train_size=0.66, random_state=0)
                mf_scores = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(single_fit)(mf, alpha, X_tr, cv) for alpha in alphas)
            else:
                 mf_scores = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(single_fit)(mf, alpha, X_tr, X_te,
                                        nested=False) for alpha in alphas)
            mf_scores = np.array(mf_scores).mean(axis=1)
            best_alpha_arg = mf_scores.argmin()
            best_alpha = alphas[best_alpha_arg]
            mf.set_params(alpha=best_alpha)

        cb = Callback(X_tr, X_te, refit=False)
        mf.set_params(callback=cb)
        mf.fit(X_tr)
        results[dict_id[mf]] = dict(name=names[dict_id[mf]],
                                    time=cb.times,
                                    rmse=cb.rmse)
        if cross_val:
            results[dict_id[mf]]['alphas'] = alphas.tolist()
            results[dict_id[mf]]['cv_alpha'] = mf_scores.tolist()
            results[dict_id[mf]]['best_alpha'] = mf.alpha

        with open(join(output_dir, 'results_%s_%s.json' % (version,
                                                           random_state)),
                  'w+') as f:
            json.dump(results, f)

        print('Done')


def single_fit(mf, alpha, X_tr, cv, nested=True):
    mf_cv = clone(mf)
    if isinstance(mf_cv, DictMF):
        mf_cv.set_params(n_epochs=2)
    else:
        mf_cv.set_params(max_iter=10)
    mf_cv.set_params(alpha=alpha)
    if nested:
        score = cross_val_score(mf_cv, X_tr, cv)
    else:
        X_te = cv
        mf_cv.fit(X_tr)
        score = [mf_cv.score(X_te)]
    return score


def plot_benchs(output_dir=expanduser('~/output/recommender/benches')):
    fig = plt.figure()

    fig.subplots_adjust(right=.9)
    fig.subplots_adjust(top=.85)
    fig.subplots_adjust(bottom=.12)
    fig.subplots_adjust(left=.08)
    fig.set_figheight(1.6)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5])

    ylims = {'100k': [.90, .96], '1m': [.865, .915], '10m': [.80, .868], 'netflix': [.93, .99]}
    xlims = {'100k': [0.0001, 10], '1m': [0.05, 10], '10m': [0.5, 100], 'netflix': [30, 3000]}

    names = {'dl_partial': 'Proposed \n(partial projection)',
             'dl': 'Proposed \n(full projection)',
             'cd': 'Coordinate descent'}
    for i, version in enumerate(['1m', '10m', 'netflix']):
        with open(join(output_dir, 'results_%s.json' % version), 'r') as f:
            data = json.load(f)
        ax_time = fig.add_subplot(gs[0, i])
        ax_time.grid()
        sns.despine(fig, ax_time)

        ax_time.spines['left'].set_color((.6, .6, .6))
        ax_time.spines['bottom'].set_color((.6, .6, .6))
        ax_time.xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax_time.yaxis.set_tick_params(color=(.6, .6, .6), which='both')

        for tick in ax_time.xaxis.get_major_ticks():
            tick.label.set_fontsize(5)
            tick.label.set_color('black')
        for tick in ax_time.yaxis.get_major_ticks():
            tick.label.set_fontsize(5)
            tick.label.set_color('black')

        if i == 0:
            ax_time.set_ylabel('RMSE on test set')
        if i == 2:
            ax_time.set_xlabel('CPU time (s)')
            ax_time.xaxis.set_label_coords(1.15, -0.05)

        ax_time.grid()
        palette = sns.color_palette('deep')
        color = {'dl_partial': palette[0], 'dl': palette[1], 'cd': palette[2]}
        for estimator in sorted(OrderedDict(data).keys()):
            this_data = data[estimator]
            ax_time.plot(this_data['time'], this_data['rmse'],
                         label=names[estimator], color=color[estimator], linewidth=2,
                         linestyle='-' if estimator != 'cd' else '--')
        if version == 'netflix':
            ax_time.legend(loc='upper left', bbox_to_anchor=(.6, 1),
                           frameon=False)
        ax_time.set_xscale('log')
        ax_time.set_ylim(ylims[version])
        ax_time.set_xlim(xlims[version])
        ax_time.set_title('Movielens %s' % version if version != 'netflix' else 'Netflix')
    plt.savefig('output.pdf')


if __name__ == '__main__':
    # main('netflix', n_jobs=1, cross_val=False)
    # main('100k', n_jobs=3, cross_val=True)
    # main('1m', cross_val=True, n_jobs=15, random_state=0)
    # main('10m', n_jobs=15, cross_val=True, random_state=0)
    # for i in range(5):
    #     main('1m', cross_val=True, n_jobs=15, random_state=i)
    #     main('10m', n_jobs=15, random_state=i, cross_val=True)
    plot_benchs()
