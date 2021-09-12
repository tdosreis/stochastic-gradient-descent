import numpy as np


class LogisticRegressionModel:
    """The Logistic Regression class. It implements a gradient descent
    strategy to update the function weights. There are two methods:

    a) the traditional method known as gradient descent computes the
    coefficients for the logistic regression in the entire dataset at once.

    b) the online learning method, known as mini-batch stochastic gradient
    descent which computes the coefficients at each iteration.

    Parameters
    ----------
    stochastic : bool, default=False
        Whether or not SGD should be used.

    batch_size : int, default=5
        Number of rows to update the weights.

    n_iter : int, default=1000
        Number of iterations for the algorithm to converge.

    eta : float, default=0.01
        The learning rate.

    fixed_weights : list, default=None
        List with the value of the fixed weights.

    fixed_mask : list, default=None
        The indices of the variables with fixed weights.

    Attributes
    ----------

    coefs_ : dictionary
        Coefficient of the features in the decision function.

    costs_ : dictionary
        Costs of every iteration.
    """

    def __init__(self,
                 stochastic=False,
                 batch_size=5,
                 n_iter=1000,
                 eta=0.01,
                 fixed_weights=None,
                 fixed_mask=None):

        self.stochastic = stochastic
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.learning_rate = eta
        self.fixed_weights = fixed_weights
        self.fixed_mask = fixed_mask
        self.coefs_ = {}
        self.costs_ = {}

    def fit(self, X, y, theta):

        if self.stochastic:
            self.coefs_opt_st, _, _ = (
                self.stochastic_grad_descent(
                    theta=theta,
                    X=X,
                    y=y,
                )
            )

        else:
            self.coefs_opt, _, _ = (
                self.grad_descent(
                    theta=theta,
                    X=X,
                    y=y,
                )
            )

    def predict(self, X):
        if self.stochastic:
            y_pred = self.sigmoid(np.dot(X, self.coefs_opt_st))
            return y_pred

        else:
            y_pred = self.sigmoid(np.dot(X, self.coefs_opt))
            return y_pred

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def cost_function(self, theta, X, y):
        m = len(y)
        y_pred = self.sigmoid(np.dot(X, theta))
        loss = (
            -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            )
        return loss

    def grad_descent(self, theta, X, y):   # entire batch training
        m = len(y)
        n = X.shape[1]
        y = y.reshape(-1, 1)
        costs_ = np.zeros((self.n_iter, 1))
        coefs_ = np.zeros((self.n_iter, n))
        for i in range(self.n_iter):
            y_pred = self.sigmoid(np.dot(X, theta))
            error = y_pred - y
            theta = theta - (1/m)*self.learning_rate*(np.dot(X.T, error))
            if self.fixed_mask:   # force fixed weights
                theta[self.fixed_mask] = (
                    np.array(self.fixed_weights).reshape(-1, 1)
                    )
            cost = self.cost_function(theta, X, y)
            self.coefs_[i] = theta.T
            self.costs_[i] = cost
        return theta, costs_, coefs_

    def stochastic_grad_descent(self, theta, X, y):   # "online learning"
        m = len(y)
        n = X.shape[1]
        y = y.reshape(-1, 1)
        costs_ = np.zeros((self.n_iter, 1))
        coefs_ = np.zeros((self.n_iter, n))
        for j in range(self.n_iter):
            cost = 0.0
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for i in range(0, m, self.batch_size):
                X_i = X[i:i+self.batch_size]
                y_i = y[i:i+self.batch_size]
                y_pred = self.sigmoid(np.dot(X_i, theta))
                error = y_pred - y_i
                theta = theta - (1/m)*self.learning_rate*(np.dot(X_i.T, error))
                if self.fixed_mask:    # force fixed weights
                    theta[self.fixed_mask] = (
                        np.array(self.fixed_weights).reshape(-1, 1)
                        )
                cost += self.cost_function(theta, X_i, y_i)
            self.coefs_[j] = theta.T
            self.costs_[j] = cost
        return theta, costs_, coefs_
