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
                    gamma=0.0, risk_free_rate=0.0, weights_current=None, c=None):
    """
    Calculate the negative (since the optimizer minimizes the objective)
    Sharpe ratio of a portfolio. Penalize deviations from current portfolio
    by an amount given by the transaction cost vector c.

    Parameters
    ----------
    weights :  np.ndarray
        weights vector of next period's portfolio
    expected_return s: pd.Series
        expected return vector
    cov_matrix : pd.DataFrame
        covariance matrix of asset returns
    gamma : float, optional
        l2 regularisation parameter, defaults to 0.0
    risk_free_rate : float, optional
        risk-free rate for Sharpe ratio calculation, defaults to 0.0
    weights_current : np.ndarray, optional
        weights vector of current portfolio (if you already hold positions)
    c : np.ndarray, optional
        vector of estimated transaction costs for each asset (typically positive)

    Returns
    -------
    neg_sharpe : float
        negative expected Sharpe ratio (after transaction costs if provided)
        and regularization

    Example:
    >>> weights = np.array([0.25]*4)
    >>> expected_returns = np.array([0.1, 0.2, -0.1, 0.0])
    >>> cov_matrix =  np.identity(4)
    >>> weights_current = np.array([0.2]*4)
    >>> c = np.array([0.025]*4)
    >>> negative_sharpe(weights, expected_returns, cov_matrix,
    ...                 gamma=0.0, risk_free_rate=0.0,
    ...                 weights_current=weights_current, c=c)
    -0.09000000000000002
    >>> negative_sharpe(weights, expected_returns, cov_matrix)
    -0.10000000000000002
    """

    mu = weights.dot(expected_returns)
    if weights_current is not None:
        if c is None:
            raise ValueError('c cannot be of type None if weights_current is provided.')
        else:
            mu -= np.dot(np.abs(weights - weights_current).T, c)
    sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    l2_reg = gamma * np.dot(weights.T, weights)
    neg_sharpe = -(mu - risk_free_rate) / sigma + l2_reg
    return neg_sharpe


def pc_cov(cov, n_components):
    """ Decopose the covariance matrix into eigenvalues and
    eigenvectors. Discard eigenvalues smaller than the n_compenent's
    eigenvalue and return the denoised covariance matrix.

    Parameters
    ----------
    cov : np.ndarray of dim 2
        covariance matrix of returns
    n_components: int
        number of principal components (>=1)

    Returns
    -------
    denoised_cov : pd.DataFrame
        denoised covariance matrix

    Example:
    >>> cov = np.array([[1, 0.5], [0.5, 0.7]])
    >>> pc_cov(cov, n_components=1).values
    array([[0.88313051, 0.65707617],
           [0.65707617, 0.48888481]])
    """
    assert n_components >= 1

    evalues, Q = np.linalg.eig(cov)
    # ensure sorted evalues
    evalues_sorted = np.sort(evalues)
    # print(evalues_sorted)
    evalues[evalues < evalues_sorted[-n_components]] = 0

    # Since cov is symmetric use spectral theorem
    Q_T = np.matrix.transpose(Q)
    L = np.diag(evalues)
    # Due to spectral theorem all eigenvalues must be real. I encountered
    # complex eigenvalues regardless. Hence, extract real part.
    denoised_cov = pd.DataFrame(Q.dot(L).dot(Q_T)).apply(np.real)
    return denoised_cov


if __name__ == "__main__":
    import doctest
    doctest.testmod()
