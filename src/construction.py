import scipy.optimize
import numpy as np
import pandas as pd


def sector_neutral_threshold(pf,
                             new_assets,
                             new_factor_value,
                             new_sector_id,
                             quantile,
                             rebate_rate=None,
                             min_rebate_rate=None,
                             threshold=None):
    """Select stocks accordig to their factor value subject to sector
    neutrality constraint. Only trade an existing position for a
    new one if the new position is significantly better than the old one.
    Exclude hard to borrwo stocks.
    Parameters
    ----------
    pf: portfolio object
        currently invested portfolio
    new_assets: pd.index, list
        index of new_factor_value
    new_factor_value: pd.Series
        current factor values of assets in universe
    new_sector_id: pd.Series
        sector or cluster identification of assets in universe
    quantile: float
        quantile of assets per industry to long/short
    rebate_rate: pd.Series
        Fed funds - shorts borrowing fee for all assets in universe
    min_rebate_rate: float
        minimum acceptable rate at which shorting is permitted (e.g -0.04)
    threshold: float
        quantile of grace space (e.g. 0.1 if asset in top 10% shouldn't
        be liquidated)

    Returns
    -------
    longs: list
        Optimal Longs to Buy
    shorts: list
        Optimal Shorts to Sell
    """

    if rebate_rate is None:
        rebate_rate = np.zeros(len(new_assets))

    factor_df = pd.DataFrame(index=new_assets,
                             data={'factor_value': new_factor_value,
                                   'sector_id': new_sector_id,
                                   'rebate_rate': rebate_rate})
    sectors = factor_df.sector_id.unique()
    short_stocks = []
    long_stocks = []

    for ind in sectors:
        ind_stocks = factor_df[factor_df.sector_id == ind].sort_values(
                                                    by=['factor_value'],
                                                    ascending=True)
        n_ind = len(ind_stocks)
        if n_ind < 2:
            continue
        n_select = int(quantile*n_ind)+1
        n_grace = int(threshold*n_ind)+1

        long_ind = list(ind_stocks.iloc[-n_select:].index)
        short_ind = list(ind_stocks[ind_stocks.rebate_rate >
                         min_rebate_rate].iloc[:n_select].index)

        # sort shorts from worst to best
        short_ind.reverse()

        if threshold is not None:
            long_grace = list(ind_stocks.iloc[-n_grace:].index)
            short_grace = list(ind_stocks[ind_stocks.rebate_rate >
                               min_rebate_rate].iloc[:n_grace].index)
            # sort shorts from worst to best
            short_grace.reverse()

            long_pos_in_grace = [x for x in list(
                set(long_grace).intersection(
                    set(pf.longs()))) if x not in long_ind]

            short_pos_in_grace = [x for x in list(
                set(short_grace).intersection(
                    set(pf.shorts()))) if x not in short_ind]

            for index_l, l in enumerate(long_pos_in_grace):
                if index_l < len(long_ind):
                    long_ind[index_l] = l
                else:
                    long_ind.append(l)

            for index_s, s in enumerate(short_pos_in_grace):
                if index_s < len(short_ind):
                    short_ind[index_s] = s
                else:
                    short_ind.append(s)
        long_stocks.extend(long_ind)
        short_stocks.extend(short_ind)

    return {'longs': long_stocks, 'shorts': short_stocks}


def optimal_non_reducing_dollar_neutral_weights(pf,
                                                n_new_longs,
                                                n_new_shorts,
                                                max_lev=1,
                                                basis_weight=0.01,
                                                max_long_weight=0.1,
                                                max_short_weight=0.1):
    """ Get optimal weights (long-short) to be dollar neutral without reducing
        current position and incurring transaction costs for that.
        Parameters
        ----------
        pf: portfolio object
            currently invested portfolio
        n_new_longs: int
            Number of Potential new Longs
        n_new_shorts: int
            Number of Potential new Shorts
        max_lev: float
            maximum permitted leverage (e.g. 1.2)
        basis_weight: float
            Default Position Weight (e.g. 0.01 for 1% of PV)
        max_long_weight: float
            Max Position Weight for Longs (e.g. 0.1 for 10% of PV)
        max_short_weight: float
            Max Position Weight for Shorts (non-negative)
            (e.g. 0.1 for 10% of PV)

        Returns
        -------
        long_weights: float
            Optimal Long Position Weights (non-negative)
        short_weights: float
            Optimal Short Position Weights (non-negative)
        success: bool
            Successful optimization
        """

    if min(max_lev, basis_weight, max_long_weight,
           max_short_weight, n_new_longs, n_new_shorts) < 0:
        raise ValueError('Parameters must be non-negative.')

    lep = pf.long_exposure_pct()
    sep = pf.short_exposure_pct()

    # if trivial solution, use basis_weights
    if (lep == 0 and sep == 0 and (n_new_longs == 0 or n_new_shorts == 0)):
        long_weights, short_weights = basis_weight, basis_weight
        success = True
    else:
        def exposusure_diff(weights):
            return np.sqrt(((lep+n_new_longs*weights[0]) -
                           (sep+n_new_shorts * weights[1]))**2)

        def leverage_con(weights):
            return max_lev - (lep + n_new_longs * weights[0] +
                              sep + n_new_shorts * weights[1])

        cons = [{'type': 'ineq', 'fun': leverage_con}, ]
        bnds = np.array([(0, max_long_weight), (0, max_short_weight)])
        x0 = np.array([basis_weight, basis_weight])
        res = scipy.optimize.minimize(exposusure_diff, x0,
                                      constraints=cons, bounds=bnds)
        long_weights, short_weights = res.x[0], res.x[1]
        success = res.success

    # make sure leverage constraint is not violated
    if (lep + n_new_longs * long_weights + sep +
       n_new_shorts * short_weights) > max_lev:
        print('leverage constraint violated')
        success = False
    return {'longs_weights': long_weights,
            'short_weights': short_weights,
            'success': success}


def dollar_neutral_top_selection(pf, signal,
                                 long_quantile=0.1, short_quantile=0.1,
                                 min_th=0, buy_low=True,
                                 long_th_incr=None, short_th_incr=None,
                                 liquidating_th=0):
    """ Select best new long and short positions such that the portfolio
    is as neutral has possible without reducing positions and being equally
    weighted. The signal is assumed to be a mean reversion signal,
    i.e., high positive values are sold and negative values are bought (if
    not set buy_low=False).
    Parameters
    ----------
    pf: portfolio object
        currently invested portfolio
    long_quantile: float
        Signal quantile of long threshold
    short_quantile: float
        Signal quantile of short threshold
    min_th: float
        No new position below this value even if within quantile.
    buy_low: bool
        True if low (large negative) signal values will lead to buy
        decision and high (large positive) signal values will lead to sell
        decision. False otherwise.
    long_th_incr: float:
        Above which treshold is adding to longs (given that their actual
        position is not yet equal to the goal position) permitted.
        None if equal to new long threshold.
    short_th_incr: float
        Above which treshold is adding to shorts (given that their actual
        position is not yet equal to the goal position) permitted.
        None if equal to new short threshold.
    liquidating_th: float
        At which signal value are positions liquidated.
        This parameter is used compute current position
        in a forward-looking fashion. Set to None to ignore
        liquidation.
    Returns
    -------
    new_longs: list
        New longs to buy
    new_shorts: list
        New shorts to sell
    long_th: float
        Signal threshold for longs
    short_th: float
        Signal threshold for shorts
    current_longs_incr: list
        current longs with signal better than long_th
    current_shorts_incr: list
        current shorts with signal better than short_th
     """
    signal = pd.Series(signal).dropna()

    if not buy_low:
        signal = -signal

    if liquidating_th is not None:
        current_longs = pf.longs().drop(signal[signal >= -liquidating_th].index,
                                        axis=0, errors='ignore').index
        current_shorts = pf.shorts().drop(signal[signal <= liquidating_th].index,
                                          axis=0, errors='ignore').index

    else:
        current_longs = pf.longs().index
        current_shorts = pf.shorts().index

    signal_ascending = signal.sort_values(ascending=True)
    signal_descending = signal.sort_values(ascending=False)

    long_th = min(
        signal_ascending.quantile(q=long_quantile,
                                  interpolation='lower'), -min_th)

    short_th = max(
        signal_descending.quantile(q=1-short_quantile,
                                   interpolation='higher'), min_th)
    if long_th_incr is None:
        long_th_incr = long_th

    if short_th_incr is None:
        short_th_incr = short_th

    new_potential_longs = signal_ascending[signal_ascending <= long_th]
    new_potential_longs.drop(current_longs, axis=0,
                             inplace=True, errors='ignore')

    new_potential_shorts = signal_descending[signal_descending >= short_th]
    new_potential_shorts.drop(current_shorts, axis=0, inplace=True,
                              errors='ignore')

    current_longs_incr = signal_ascending[
                signal_ascending <= long_th_incr].index.intersection(
                                                    current_longs).tolist()

    current_shorts_incr = signal_descending[
                signal_descending >= short_th_incr].index.intersection(
                                                current_shorts).tolist()

    if (len(new_potential_longs) +
        len(current_longs)) >= (len(new_potential_shorts) +
                                len(current_shorts)):
        new_shorts = new_potential_shorts.index.tolist()
        n_longs = max(0, len(new_potential_shorts) +
                      len(current_shorts) - len(current_longs))
        new_longs = new_potential_longs.iloc[:n_longs].index.tolist()
    else:
        new_longs = new_potential_longs.index.tolist()
        n_shorts = max(0, len(new_potential_longs) +
                       len(current_longs) - len(current_shorts))
        new_shorts = new_potential_shorts.iloc[:n_shorts].index.tolist()

    if not buy_low:
        long_th = -long_th
        short_th = -short_th

    return {'new_longs': new_longs, 'new_shorts': new_shorts,
            'long_th': long_th, 'short_th': short_th,
            'current_longs_incr': current_longs_incr,
            'current_shorts_incr': current_shorts_incr}
