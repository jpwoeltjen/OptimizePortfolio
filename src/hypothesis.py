import numpy as np
from scipy import stats


def jobson_korkie(returns_1, returns_2):
    """ Tests the null H0: sharpe_1 == sharpe_2 if returns
    are multivariate Gaussian.

    Parameters
        ----------
        returns_1 : np.array, pd.Series, list
            Assets in Portfolio
        returns_2 : np.array, pd.Series, list
    Returns
    -------
    dict
        {'p_value': p_value, 't_statistic': t_stat}"""

    returns_1 = np.array(returns_1)
    returns_2 = np.array(returns_2)
    sharpe_1 = np.mean(returns_1)/np.std(returns_1)
    sharpe_2 = np.mean(returns_2)/np.std(returns_2)
    corr = np.corrcoef(returns_1, returns_2)[1, 0]
    n = min(returns_1.shape[0], returns_2.shape[0])
    asyvar = ((2*(1-corr) +
              1/2*(sharpe_1**2+sharpe_2**2-2*sharpe_1*sharpe_2*corr**2)) / n)
    t_stat = (sharpe_1 - sharpe_2)/np.sqrt(asyvar)
    p_value = stats.t.sf(np.abs(t_stat), n-1)*2

    return {'p_value': p_value, 't_statistic': t_stat}
