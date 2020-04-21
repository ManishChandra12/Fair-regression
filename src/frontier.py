import os
import random
import argparse
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
import cvxpy as cp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from src.loss import logloss, logloss_sep, l2norm, get_loss
from src.fairness_penalty import fairness_penalty
from CONSTANTS import PROCESSED_DATA_DIR, ROOT_DIR

np.random.seed(10)
random.seed(10)


def error(y_true, X, w):
    """
    Computes the MSE between the true and predicted class labels in single model setting
    :param y_true: ground-truth labels
    :param X: features (data points)
    :param w: learned weight vector
    :return: MSE between the true and predicted class labels
    """
    y_pred = 1 / (1 + np.exp(-X @ w))
    return mean_squared_error(y_true, y_pred)


def error_sep(indices1, indices2, y_true, X, w1, w2):
    """
    Computes the MSE between the true and predicted class labels in separate model setting
    :param indices1: the indices of X that fall in protected group 1
    :param indices2: the indices of X that fall in protected group 2
    :param y_true: ground-truth labels
    :param X: features (data points)
    :param w1: learned weight vector for group 1
    :param w2: learned weight vector for group 2
    :return:
    """
    y_pred = (indices1.reshape(indices1.shape[0], 1) * (1 / (1 + np.exp(-X @ w1)))) + (indices2.reshape(indices2.shape[0], 1) * (1 / (1 + np.exp(-X @ w2))))
    return mean_squared_error(y_true, y_pred)


def main(X, y, idx1, idx2, gamma_vals, lambda_vals, proc, dataset):
    # stack a column of 1s
    X = np.c_[np.ones(len(X)), X]

    MSEs = {'individual': [], 'group': [], 'hybrid': [], 'individualsep': [], 'groupsep': [], 'hybridsep': []}
    FPs = {'individual': [], 'group': [], 'hybrid': [], 'individualsep': [], 'groupsep': [], 'hybridsep': []}
    lambd = cp.Parameter(nonneg=True)
    for lmd in lambda_vals:
        print("Processing lambda = " + str(lmd) + " ...")
        pool = Pool(processes=proc)
        # get loss values for this lambda and all the gammas
        gamma_individual = gamma_group = gamma_hybrid = gamma_individualsep = gamma_groupsep = gamma_hybridsep = None
        if lmd != 'inf':
            loss_values = pool.starmap(get_loss, (itertools.product([X], [y], [idx1], [idx2], gamma_vals, [lmd])))

            # choose the best gamma value for this lambda
            gamma_individual = gamma_vals[np.argmin([i[0] for i in loss_values], axis=0)]
            gamma_group = gamma_vals[np.argmin([i[1] for i in loss_values], axis=0)]
            gamma_hybrid = gamma_vals[np.argmin([i[2] for i in loss_values], axis=0)]
            gamma_individualsep = gamma_vals[np.argmin([i[3] for i in loss_values], axis=0)]
            gamma_groupsep = gamma_vals[np.argmin([i[4] for i in loss_values], axis=0)]
            gamma_hybridsep = gamma_vals[np.argmin([i[5] for i in loss_values], axis=0)]

            # Once optimal gamma is found, solve the objective function for this particular lambda
            lambd.value = lmd

        # use 10-fold cross-validation
        cv = KFold(n_splits=10, shuffle=False)
        MSE = [0, 0, 0, 0, 0, 0]  # mean-squared errors
        FP = [0, 0, 0, 0, 0, 0]  # fairness penalties
        for train_index, test_index in cv.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            w1 = cp.Variable((X_train.shape[1], 1))
            w2 = cp.Variable((X_train.shape[1], 1))
            w3 = cp.Variable((X_train.shape[1], 1))
            w1sep1 = cp.Variable((X_train.shape[1], 1))
            w1sep2 = cp.Variable((X_train.shape[1], 1))
            w2sep1 = cp.Variable((X_train.shape[1], 1))
            w2sep2 = cp.Variable((X_train.shape[1], 1))
            w3sep1 = cp.Variable((X_train.shape[1], 1))
            w3sep2 = cp.Variable((X_train.shape[1], 1))
            penal = fairness_penalty(idx1[train_index], idx2[train_index], w1, w2, w3, w1sep1, w1sep2, w2sep1, w2sep2, w3sep1, w3sep2, X_train, y_train)

            if lmd != 'inf':
                problem_individual = cp.Problem(cp.Minimize(logloss(X_train, y_train, w1) + lambd * penal['individual'] + gamma_individual * l2norm(w1)))
                problem_group = cp.Problem(cp.Minimize(logloss(X_train, y_train, w2) + lambd * penal['group'] + gamma_group * l2norm(w2)))
                problem_hybrid = cp.Problem(cp.Minimize(logloss(X_train, y_train, w3) + lambd * penal['hybrid'] + gamma_hybrid * l2norm(w3)))
                problem_individualsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w1sep1, w1sep2) + lambd * penal['individualsep'] + gamma_individualsep * (l2norm(w1sep1) + l2norm(w1sep2))))
                problem_groupsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w2sep1, w2sep2) + lambd * penal['groupsep'] + gamma_groupsep * (l2norm(w2sep1) + l2norm(w2sep2))))
                problem_hybridsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w3sep1, w3sep2) + lambd * penal['hybridsep'] + gamma_hybridsep * (l2norm(w3sep1) + l2norm(w3sep2))))
            else:
                problem_individual = cp.Problem(cp.Minimize(penal['individual']))
                problem_group = cp.Problem(cp.Minimize(penal['group']))
                problem_hybrid = cp.Problem(cp.Minimize(penal['hybrid']))
                problem_individualsep = cp.Problem(cp.Minimize(penal['individualsep']))
                problem_groupsep = cp.Problem(cp.Minimize(penal['groupsep']))
                problem_hybridsep = cp.Problem(cp.Minimize(penal['hybridsep']))

            problem_individual.solve()
            # print(problem.status)
            MSE[0] += error(y_test, X_test, w1.value)

            problem_group.solve()
            # print(problem.status)
            MSE[1] += error(y_test, X_test, w2.value)

            problem_hybrid.solve()
            # print(problem.status)
            MSE[2] += error(y_test, X_test, w3.value)

            problem_individualsep.solve()
            MSE[3] += error_sep(idx1[test_index], idx2[test_index], y_test, X_test, w1sep1.value, w1sep2.value)

            problem_groupsep.solve()
            MSE[4] += error_sep(idx1[test_index], idx2[test_index], y_test, X_test, w2sep1.value, w2sep2.value)

            problem_hybridsep.solve()
            MSE[5] += error_sep(idx1[test_index], idx2[test_index], y_test, X_test, w3sep1.value, w3sep2.value)

            # get fairness penalties on test set
            ps = fairness_penalty(idx1[test_index], idx2[test_index], w1.value, w2.value, w3.value, w1sep1.value, w1sep2.value, w2sep1.value, w2sep2.value, w3sep1.value, w3sep2.value, X_test, y_test, 1)
            FP[0] += (ps['individual']).value
            FP[1] += (ps['group']).value
            FP[2] += (ps['hybrid']).value
            FP[3] += (ps['individualsep']).value
            FP[4] += (ps['groupsep']).value
            FP[5] += (ps['hybridsep']).value

        # store the MSEs and FPs for plotting
        MSEs['individual'].append(MSE[0] / 10)
        FPs['individual'].append(FP[0] / 10)
        MSEs['group'].append(MSE[1] / 10)
        FPs['group'].append(FP[1] / 10)
        MSEs['hybrid'].append(MSE[2] / 10)
        FPs['hybrid'].append(FP[2] / 10)
        MSEs['individualsep'].append(MSE[3] / 10)
        FPs['individualsep'].append(FP[3] / 10)
        MSEs['groupsep'].append(MSE[4] / 10)
        FPs['groupsep'].append(FP[4] / 10)
        MSEs['hybridsep'].append(MSE[5] / 10)
        FPs['hybridsep'].append(FP[5] / 10)

        if lmd != 'inf':
            print("Individual-single model: best_gamma={}, MSE={}, FP={}".format(gamma_individual, MSE[0] / 10, FP[0] / 10))
            print("Group-single model: best_gamma={}, MSE={}, FP={}".format(gamma_group, MSE[1] / 10, FP[1] / 10))
            print("Hybrid-single model: best_gamma={}, MSE={}, FP={}".format(gamma_hybrid, MSE[2] / 10, FP[2] / 10))
            print("Individual-separate model: best_gamma={}, MSE={}, FP={}".format(gamma_individual, MSE[3] / 10, FP[3] / 10))
            print("Group-separate model: best_gamma={}, MSE={}, FP={}".format(gamma_group, MSE[4] / 10, FP[4] / 10))
            print("Hybrid-separate model: best_gamma={}, MSE={}, FP={}".format(gamma_hybrid, MSE[5] / 10, FP[5] / 10))
            print()
    plt.xlim(right=0.1)  # to replicate the result obtained by authors
    ind_sin, = plt.plot(FPs['individual'], MSEs['individual'], label='Individual, single')
    grp_sin, = plt.plot(FPs['group'], MSEs['group'], label='Group, single')
    hyb_sin, = plt.plot(FPs['hybrid'], MSEs['hybrid'], label='Hybrid, single')
    ind_sep, = plt.plot(FPs['individualsep'], MSEs['individualsep'], label='Individual, separate')
    grp_sep, = plt.plot(FPs['groupsep'], MSEs['groupsep'], label='Group, separate')
    hyb_sep, = plt.plot(FPs['hybridsep'], MSEs['hybridsep'], label='Hybrid, separate')
    plt.legend([ind_sin, grp_sin, hyb_sin, ind_sep, grp_sep, hyb_sep], ['Individual, single', 'Group, single', 'Hybrid, single', 'Individual, separate', 'Group, separate', 'Hybrid, separate'])
    plt.xlabel('Fairness Loss')
    plt.ylabel('MSE')
    plt.title(dataset.upper())
    # plt.show()
    plt.savefig(os.path.join(ROOT_DIR, "output", dataset+".png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['compas', 'adult', 'communities', 'default', 'lawschool'], required=True, help="dataset to use")
    parser.add_argument("--proc", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], default=7, help="number of processes to run in parallel (between 1 and 16)")
    args = parser.parse_args()

    # gamma values to try
    trials = 7
    gamma_vals = np.linspace(0.01, 1, trials)

    # lambda values to feed
    lambda_vals = [0, 0.1, 1, 50, 'inf']

    proc = args.proc

    data = X = y = idx1 = idx2 = None
    if args.dataset == 'default' :
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'Default/default_processed.csv'), index_col=None)
        y = data[['default_payment_next_month']].values
        X = data.drop(['default_payment_next_month', 'SEX_MALE','SEX_FEMALE'], axis=1).values
        # indices denoting which row is from which protected group
        idx1 = (data['SEX_MALE'] == 1).values
        idx2 = (data['SEX_FEMALE'] == 1).values
    main(X, y, idx1, idx2, gamma_vals, lambda_vals, proc, args.dataset)
