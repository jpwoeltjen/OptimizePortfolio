import scipy.optimize
import numpy as np
import pandas as pd


class Portfolio:
    """ Optimize position selection and weights subject to leverage,
     position concentration, transaction cost and sector exposure constraints.
    """
    def __init__(self, assets, position, price,
                 factor_value=None, sector_id=None):
        """
        Parameters
        ----------
        assets : array
            Assets in Portfolio
        position : array
            Current Long (non-negative) and
            Short (non-positive) Positions of 'assets' in Shares
        price : array
            Current Price of 'assets' in Dollars
        factor_value : array
            Factor values of 'assets'
        sector_id : array
            Sector identification of 'assets'
        """

        if factor_value is None:
            factor_value = np.ones(len(assets))

        if sector_id is None:
            sector_id = np.zeros(len(assets))

        self.portfolio_df = pd.DataFrame(index=assets,
                                         data={'position': position,
                                               'price': price,
                                               'factor_value': factor_value,
                                               'sector_id': sector_id})

    @classmethod
    def from_dataframe(cls, dataframe):
        """Alternative constructor from an asset indexed pd.DataFrame()"""
        asset = dataframe.index
        position = dataframe.iloc[:, 0]
        price = dataframe.iloc[:, 1]
        factor_value = dataframe.iloc[:, 2]
        sector_id = dataframe.iloc[:, 3]
        return cls(asset, position, price, factor_value, sector_id)

    def longs(self):
        """Returns positions with positive dollar value."""
        return self.portfolio_df[self.portfolio_df.position > 0]

    def shorts(self):
        """Returns positions with negative dollar value."""
        return self.portfolio_df[self.portfolio_df.position < 0]

    def long_exposure(self):
        """Dollar value of longs"""
        longs = self.longs()
        return (longs.position * longs.price).sum()

    def short_exposure(self):
        """Dollar value of shorts"""
        shorts = self.shorts()
        return -(shorts.position * shorts.price).sum()

    def gross_exposure(self):
        """Dollar gross exposure"""
        return self.long_exposure() + self.short_exposure()

    def net_exposure(self):
        """Dollar net exposure"""
        return self.long_exposure() - self.short_exposure()

    def long_exposure_pct(self):
        """Long exposure as a proportion of gross exposure"""
        return self.long_exposure() / self.gross_exposure()

    def short_exposure_pct(self):
        """Short exposure as a proportion of gross exposure"""
        return self.short_exposure() / self.gross_exposure()

    def sector_net_exposures(self):
        """Dollar net exposure per sector"""
        position_value = pd.DataFrame(self.portfolio_df.sector_id)
        position_value['net_exposures'] = (self.portfolio_df.position *
                                           self.portfolio_df.price)
        return position_value.groupby(['sector_id']).sum()

    def position(self, asset):
        if asset in self.portfolio_df.index:
            return self.portfolio_df.position[asset]
        else:
            return 0

    def trade(self, asset, amount, price,
              factor_value=None, sector_id=None):
        """Positive amount: buying asset to cover short or
        create new position.
        Negative amount: selling asset to close long or
        create new short position.
        """
        if asset in self.portfolio_df.index:
            self.portfolio_df.loc[asset, 'position'] += amount
            if self.portfolio_df['position'][asset] == 0:
                self.portfolio_df.drop([asset], axis=0, inplace=True)
            else:
                self.portfolio_df.loc[asset, 'price'] = price
                if factor_value is not None:
                    self.portfolio_df.loc[asset, 'factor_value'] = factor_value
                if sector_id is not None:
                    self.portfolio_df.loc[asset, 'sector_id'] = sector_id

        else:
            new_position = pd.DataFrame(index=[asset],
                                        data={'position': [amount],
                                              'price': [price],
                                              'factor_value': [factor_value],
                                              'sector_id': [sector_id]})

            self.portfolio_df = self.portfolio_df.append(new_position,
                                                         verify_integrity=True)

    def __repr__(self):
        msg = f"Portfolio(assets, position, price, factor_value, sector_id)"
        return msg

    def __str__(self):
        msg = (f"Portfolio: {len(self.longs()) + len(self.shorts())} " +
               f"Assets, {self.long_exposure()}$ Long, " +
               f"{self.short_exposure()}$ Short")
        return msg

    def optimal_non_reducing_dollar_neutral_weights(self,
                                                    n_new_longs,
                                                    n_new_shorts,
                                                    max_lev=1,
                                                    basis_weight=0.01,
                                                    max_long_weight=0.1,
                                                    max_short_weight=0.1):
        """ Get optimal weights (long-short) to be dollar neutral without reducing
            current position and incurring transaction costs for that.
            basis_weight : float
            Default Position Weight (e.g. 0.01 for 1% of PV)
            max_long_weight : float
                Max Position Weight for Longs (e.g. 0.1 for 10% of PV)
            max_short_weight : float
                Max Position Weight for Shorts (non-negative)
                (e.g. 0.1 for 10% of PV)
            n_new_longs : int
                Number of Potential new Longs
            n_new_shorts : int
                Number of Potential new Shorts

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

        lep = self.long_exposure_pct()
        sep = self.short_exposure_pct()

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

    def threshold_sector_neutral_positions(self,
                                           new_assets,
                                           new_factor_value,
                                           new_sector_id,
                                           quantile,
                                           mkt_cap=None,
                                           shorts_mkt_cap_floor=None,
                                           threshold=None):
        """Select stocks accordig to their factor value subject to sector
        neutrality constraint. Only trade an existing position for a
        new one if the new position is significantly better than the old one.
        Small cap stocks are hard to borrow: shorts_mkt_cap_floor specifies the
        minimum mkt_cap for potential shorts.

        Returns
        -------
        longs: list
            Optimal Longs to Buy
        shorts: list
            Optimal Shorts to Sell
        """

        if mkt_cap is None:
            mkt_cap = np.zeros(len(new_assets))

        factor_df = pd.DataFrame(index=new_assets,
                                 data={'factor_value': new_factor_value,
                                       'sector_id': new_sector_id,
                                       'mkt_cap': mkt_cap})
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
            short_ind = list(ind_stocks[ind_stocks.mkt_cap >
                             shorts_mkt_cap_floor].iloc[:n_select].index)

            # sort shorts from worst to best
            short_ind.reverse()

            if threshold is not None:
                long_grace = list(ind_stocks.iloc[-n_grace:].index)
                short_grace = list(ind_stocks[ind_stocks.mkt_cap >
                                   shorts_mkt_cap_floor].iloc[:n_grace].index)
                # sort shorts from worst to best
                short_grace.reverse()

                long_pos_in_grace = [x for x in list(
                    set(long_grace).intersection(
                        set(self.longs()))) if x not in long_ind]

                short_pos_in_grace = [x for x in list(
                    set(short_grace).intersection(
                        set(self.shorts()))) if x not in short_ind]

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

    def dollar_neutral_top_selection(self, signal,
                                     long_quantile=0.1, short_quantile=0.1,
                                     min_th=0, buy_low=True,
                                     long_th_incr=None, short_th_incr=None,
                                     forward_looking=True):
        """ Select best new long and short positions such that the portfolio
        is as neutral has possible without reducing positions and being equally
        weighted. The signal is assumed to be a mean reversion signal,
        i.e., high positive values are sold and negative values are bought.

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

        if forward_looking:
            current_longs = self.longs().drop(signal[signal >= 0].index,
                                              axis=0, errors='ignore').index
            current_shorts = self.shorts().drop(signal[signal <= 0].index,
                                                axis=0, errors='ignore').index

        else:
            current_longs = self.longs().index
            current_shorts = self.shorts().index

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
