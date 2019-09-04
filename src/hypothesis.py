import numpy as np
from scipy import stats


def jobson_korkie(returns_1, returns_2):
    """ Tests the null H0: sharpe_1 == sharpe_2 if returns
    are multivariate Gaussian.

    Parameters
        ----------
    returns_1: np.array, pd.Series, list
        returns of first portfolio
    returns_2: np.array, pd.Series, list
        returns of second portfolio
    Returns
    -------
    dict
        {'p_value': p_value, 't_statistic': t_stat}

    Example:
    >>> np.random.seed(1)
    >>> ret = np.random.multivariate_normal(
    ...     mean=[1, 1], cov=np.array([[1, 0.5],[0.5, 1]]), size=100)
    >>> jobson_korkie(ret[:, 0], ret[:, 1])
    {'p_value': 0.3284693132518087, 't_statistic': 0.9820495880707653}
    """

    returns_1 = np.array(returns_1)
    returns_2 = np.array(returns_2)
    if returns_1.shape != returns_1.shape:
        raise ValueError('returns_1 and returns_2 should have same dimension.')

    sharpe_1 = np.mean(returns_1)/np.std(returns_1)
    sharpe_2 = np.mean(returns_2)/np.std(returns_2)
    corr = np.corrcoef(returns_1, returns_2)[1, 0]
    n = returns_1.shape[0]
    asyvar = ((2*(1-corr) +
              1/2*(sharpe_1**2+sharpe_2**2-2*sharpe_1*sharpe_2*corr**2)) / n)
    t_stat = (sharpe_1 - sharpe_2)/np.sqrt(asyvar)
    p_value = stats.t.sf(np.abs(t_stat), n-1)*2

    return {'p_value': p_value, 't_statistic': t_stat}


def ledoit_wolf_bootstrap(returns_1, returns_2, n_bootstraps, alpha=0.05, b=1):
    """ Tests the null H0: sharpe_1 == sharpe_2 if returns
    are (1) iid non-Gaussian b=1 or (2) non-iid b>1.

    Parameters
        ----------
    returns_1: np.array, pd.Series, list
        returns of first portfolio
    returns_2: np.array, pd.Series, list
        returns of second portfolio
    n_bootstraps: int
        number of bootstrap samples
    alpha: float
        alpha level defaults to 0.05
    Returns
    -------
    dict
        {'p_value': p_value, 'ci': [conf_lower, conf_upper]}

    Example:
    >>> np.random.seed(1)
    >>> ret = np.random.multivariate_normal(
    ...     mean=[1, 1], cov=np.array([[1, 0.5], [0.5, 1]]), size=100)
    >>> ledoit_wolf_bootstrap(ret[:, 0],
    ...                       ret[:, 1],
    ...                       n_bootstraps=1000,
    ...                       alpha=0.05)
    {'p_value': 0.38461538461538464, 'ci': [-0.10648183140118805, 0.3723297322421312]}
    """

    returns_1 = np.array(returns_1)
    returns_2 = np.array(returns_2)

    if returns_1.shape != returns_1.shape:
        raise ValueError('returns_1 and returns_2 should have same dimension.')

    n = returns_1.shape[0]

    def sample_stats(returns_1, returns_2):
        n = returns_1.shape[0]
        mu_1 = np.mean(returns_1)
        mu_2 = np.mean(returns_2)
        std_1 = np.std(returns_1)
        std_2 = np.std(returns_2)
        sharpe_1 = mu_1/std_1
        sharpe_2 = mu_2/std_2

        delta = np.array([1/std_1, -1/std_2,
                         -mu_1/(2*std_1**3), mu_2/(2*std_2**3)]).reshape(1, 4)
        g = np.array([returns_1 - mu_1,
                      returns_2 - mu_2,
                      (returns_1 - mu_1)**2 - std_1**2,
                      (returns_2 - mu_2)**2 - std_2**2])
        v = np.cov(g)
        sigma = np.sqrt(delta @ v @ np.transpose(delta)*1/n)
        return sharpe_1, sharpe_2, sigma

    sharpe_1_orig, sharpe_2_orig, sigma_orig = sample_stats(returns_1,
                                                            returns_2)
    diff = sharpe_1_orig - sharpe_2_orig
    t_stat = diff / sigma_orig
    bs_t_stats = np.zeros(n_bootstraps)
    if b == 1:
        for bs in range(n_bootstraps):
            bs_returns_1 = np.random.choice(returns_1, size=n, replace=True)
            bs_returns_2 = np.random.choice(returns_2, size=n, replace=True)
            sharpe_1, sharpe_2, sigma = sample_stats(bs_returns_1,
                                                     bs_returns_2)
            bs_t_stats[bs] = ((sharpe_1 - sharpe_2) - diff) / sigma
    # TODO: Taking advantage of block structure in cov estimation
    # Ledoit and Wolf (2008) section 3.2.2
    else:
        for bs in range(n_bootstraps):
            index_start = np.random.randint(0, n, n//b)
            index_stop = index_start + b
            index_stop = index_stop - (index_stop > (n)) * (n)
            bs_returns_1 = np.zeros(n)
            bs_returns_2 = np.zeros(n)
            for block in range(n//b):
                if index_start[block] < index_stop[block]:
                    indeces = list(range(index_start[block],
                                         index_stop[block]))
                else:
                    indeces = (list(range(0, index_stop[block])) +
                               list(range(index_start[block], n)))
                bs_returns_1[block*b:(block+1)*b] = returns_1[indeces]
                bs_returns_2[block*b:(block+1)*b] = returns_2[indeces]
            sharpe_1, sharpe_2, sigma = sample_stats(bs_returns_1,
                                                     bs_returns_2)

            bs_t_stats[bs] = ((sharpe_1 - sharpe_2) - diff) / sigma

    z = np.quantile(bs_t_stats, (1-alpha))
    conf_lower = (diff - z * sigma_orig)[0][0]
    conf_upper = (diff + z * sigma_orig)[0][0]
    p_value = ((np.sum((np.abs(bs_t_stats)-np.abs(t_stat)) >= 0) + 1) /
               (n_bootstraps+1))
    return {'p_value': p_value, 'ci': [conf_lower, conf_upper]}


if __name__ == "__main__":
    import doctest
    doctest.testmod()
