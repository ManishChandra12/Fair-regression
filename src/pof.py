import numpy as np
import pandas as pd
import argparse
import os
import cvxpy as cp
import matplotlib.pyplot as plt
import random
from src.loss import logloss, logloss_sep
from src.fairness_penalty import fairness_penalty
from CONSTANTS import PROCESSED_DATA_DIR, ROOT_DIR

random.seed(10)
np.random.seed(10)


def main(X, y, idx1, idx2, dataset):
    w = cp.Variable((X.shape[1], 1))
    problem = cp.Problem(cp.Minimize(logloss(X, y, w)))
    problem.solve()

    lp_wstar = problem.value
    fp_wstar = fairness_penalty(idx1, idx2, w, w, w, w, w, w, w, w, w, X, y, 1)

    pof = {'individual': [], 'group': [], 'hybrid': [], 'individualsep': [], 'groupsep': [], 'hybridsep': []}
    alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01]

    for alpha in alphas:
        w1 = cp.Variable((X.shape[1], 1))
        w2 = cp.Variable((X.shape[1], 1))
        w3 = cp.Variable((X.shape[1], 1))
        w1sep1 = cp.Variable((X.shape[1], 1))
        w1sep2 = cp.Variable((X.shape[1], 1))
        w2sep1 = cp.Variable((X.shape[1], 1))
        w2sep2 = cp.Variable((X.shape[1], 1))
        w3sep1 = cp.Variable((X.shape[1], 1))
        w3sep2 = cp.Variable((X.shape[1], 1))
        fp = fairness_penalty(idx1, idx2, w1, w2, w3, w1sep1, w1sep2, w2sep1, w2sep2, w3sep1, w3sep2, X, y, 0)

        constraints = [fp['individual'] <= (alpha * (fp_wstar['individual']).value)]
        problem = cp.Problem(cp.Minimize(logloss(X, y, w1)), constraints)
        problem.solve()
        pof['individual'].append(problem.value / lp_wstar)

        constraints = [fp['group'] <= (alpha * (fp_wstar['group']).value)]
        problem = cp.Problem(cp.Minimize(logloss(X, y, w2)), constraints)
        problem.solve()
        pof['group'].append(problem.value / lp_wstar)

        constraints = [fp['hybrid'] <= (alpha * (fp_wstar['hybrid']).value)]
        problem = cp.Problem(cp.Minimize(logloss(X, y, w3)), constraints)
        problem.solve()
        pof['hybrid'].append(problem.value / lp_wstar)

        constraints = [fp['individualsep'] <= (alpha * (fp_wstar['individualsep']).value)]
        problem = cp.Problem(cp.Minimize(logloss_sep(idx1, idx2, X, y, w1sep1, w1sep2)), constraints)
        problem.solve()
        pof['individualsep'].append(problem.value / lp_wstar)

        constraints = [fp['groupsep'] <= (alpha * (fp_wstar['groupsep']).value)]
        problem = cp.Problem(cp.Minimize(logloss_sep(idx1, idx2, X, y, w2sep1, w2sep2)), constraints)
        problem.solve()
        pof['groupsep'].append(problem.value / lp_wstar)

        constraints = [fp['hybridsep'] <= (alpha * (fp_wstar['hybridsep']).value)]
        problem = cp.Problem(cp.Minimize(logloss_sep(idx1, idx2, X, y, w3sep1, w3sep2)), constraints)
        problem.solve()
        pof['hybridsep'].append(problem.value / lp_wstar)

    x = np.arange(len(alphas))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots()

    _ = ax.bar(x - 2 * width, pof['individual'], width, label='Individual, single')
    _ = ax.bar(x - width, pof['group'], width, label='Group, single')
    _ = ax.bar(x, pof['hybrid'], width, label='Hybrid, single')
    _ = ax.bar(x + width, pof['individualsep'], width, label='Individual, separate')
    _ = ax.bar(x + 2 * width, pof['groupsep'], width, label='Group, separate')
    _ = ax.bar(x + 3 * width, pof['hybridsep'], width, label='Hybrid, separate')

    ax.set_ylabel('Price of Fairness')
    ax.set_title('alpha')
    ax.set_xticks(x)
    ax.set_xticklabels(alphas)
    ax.set_ylim([0, 3])
    ax.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(ROOT_DIR, "output", dataset + "_pof.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['compas', 'adult', 'communities', 'default', 'lawschool'], required=True, help="dataset to use")
    args = parser.parse_args()

    data = X = y = idx1 = idx2 = None
    if args.dataset == 'compas':
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'COMPAS/compas_processed.csv'), index_col=None)
        y = data[['is_violent_recid']].values
        X = data.drop(['is_violent_recid', 'African-American', 'Caucasian'], axis=1).values
        # indices denoting which row is from which protected group
        idx1 = (data['African-American'] == 1).values
        idx2 = (data['Caucasian'] == 1).values
    main(X, y, idx1, idx2, args.dataset)

