import numpy as np
import pandas as pd
import scipy.optimize as sco
import warnings


def minimize_objective(tickers, objective_function,
                       market_neutral=True, bounds=(-1.0, 1.0), *args):

    n_assets = len(tickers)

    if market_neutral:
        if bounds[0] >= 0:
            raise ValueError("If market_neutral=True, lower bound must be negative.")
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x)}]
    else:
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    result = sco.minimize(
        objective_function,
        x0=np.array([1.0/n_assets] * n_assets),
        args=args,
        method="SLSQP",
        bounds=[bounds]*n_assets,
        constraints=constraints,
    )

    if not result["success"]:
        warnings.warn("Optimizer did not converge.")

    weights = result["x"]
    return dict(zip(tickers, weights))


def negative_sharpe(weights, expected_returns, cov_matrix,
                    gamma=0.0, risk_free_rate=0.0):
    """
    https://github.com/robertmartin8/PyPortfolioOpt/blob/master/pypfopt/objective_functions.py
    Calculate the negative Sharpe ratio of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :param cov_matrix: the covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :param gamma: L2 regularisation parameter, defaults to 0.
                    Increase if you want more non-negligible weights
    :type gamma: float, optional
    :param risk_free_rate: risk-free rate of borrowing/lending,
                     defaults to 0.02
    :type risk_free_rate: float, optional
    :return: negative Sharpe ratio
    :rtype: float
    """
    mu = weights.dot(expected_returns)
    sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    L2_reg = gamma * (weights ** 2).sum()
    return -(mu - risk_free_rate) / sigma + L2_reg


def pc_cov(cov, n_components):
    """ Decopose the covariance matrix into eigenvalues and
    eigenvectors. Discard eigenvalues smaller than the n_compenent's
    eigenvalue and return the denoised covariance matrix.
    """
    evalues, Q = np.linalg.eig(cov)
    evalues[evalues < evalues[n_components]] = 0

    # Since cov is symmetric, by spectral theorem Q_inv = Q_T
    Q_T = np.matrix.transpose(Q)
    L = np.diag(evalues)
    # Due to spectral theorem all eigenvalues must be real. I encountered
    # complex eigenvalues regardless. Hence, extract real part.
    denoised_cov = pd.DataFrame(Q.dot(L).dot(Q_T)).apply(np.real)
    return denoised_cov


if __name__ == '__main__':
    from scipy.stats import random_correlation

    expected_returns = pd.Series([1, 1, -1, -1], index=['a', 'b', 'c', 'd'])
    covar = random_correlation.rvs((.5, .8, 1.2, 1.5))

    weights = minimize_objective(expected_returns.index,
                                 negative_sharpe,
                                 True,
                                 (-1, 1),
                                 expected_returns, covar,
                                 0.0, 0.0,)
    print(weights)
