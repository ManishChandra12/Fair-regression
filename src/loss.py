from sklearn.model_selection import KFold
import cvxpy as cp
from src.fairness_penalty import fairness_penalty


def logloss(X, y, w):
    """
    Computes the logloss for logistic regression
    :param X: featuers
    :param y: ground-truth labels
    :param w: regression weight vector
    :return: logloss for logistic regression
    """

    return -(1 / X.shape[0]) * cp.sum(cp.multiply(y, -cp.logistic(-X @ w)) + cp.multiply((1 - y), -cp.logistic(X @ w)))


def l2norm(w):
    """
    Computes l2-norm of weight vector
    :param w: regression weight vector
    :return: l2-norm of w
    """

    return (cp.norm(w, 2))**2


def get_loss(X, y, idx1, idx2, g, ll):
    """
    Computes the total loss values for the given value of lambda
    :param X: pandas dataframe containing the features
    :param y: pandas dataframe containing the ground-truth labels
    :param idx1: the indices of X that fall in protected group 1
    :param idx2: the indices of X that fall in protected group 2
    :param g: value of gamma
    :param ll: value of lambda
    :return:
    """

    L = [0, 0, 0]
    lambd = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)
    # 10-fold cross validation for picking best gamma for given lambda
    cv = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        w1 = cp.Variable((X_train.shape[1], 1))
        w2 = cp.Variable((X_train.shape[1], 1))
        w3 = cp.Variable((X_train.shape[1], 1))
        penal = fairness_penalty(idx1[train_index], idx2[train_index], w1, w2, w3, X_train, y_train)

        # minimize the objective function for individual fairness - single model
        problem_individual = cp.Problem(cp.Minimize(logloss(X_train, y_train, w1) + lambd * penal['individual'] + gamma * l2norm(w1)))        
        lambd.value = ll
        gamma.value = g
        problem_individual.solve()

        # minimize the objective function for group fairness - single model
        problem_group = cp.Problem(cp.Minimize(logloss(X_train, y_train, w2) + lambd * penal['group'] + gamma * l2norm(w2)))
        lambd.value = ll
        gamma.value = g
        problem_group.solve()

        # minimize the objective function for hybrid fairness - single model
        problem_hybrid = cp.Problem(cp.Minimize(logloss(X_train, y_train, w3) + lambd * penal['hybrid'] + gamma * l2norm(w3)))
        lambd.value = ll
        gamma.value = g
        problem_hybrid.solve()

        # get loss values in test set
        floss = fairness_penalty(idx1[test_index], idx2[test_index], w1.value, w2.value, w3.value, X_test, y_test)
        L[0] += (logloss(X_test, y_test, w1.value)).value + lambd.value * (floss['individual']).value
        L[1] += (logloss(X_test, y_test, w2.value)).value + lambd.value * (floss['group']).value
        L[2] += (logloss(X_test, y_test, w3.value)).value + lambd.value * (floss['hybrid']).value
    return L
