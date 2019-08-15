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
        assets : np.array, pd.Series, list
            Assets in Portfolio
        position : np.array, pd.Series, list
            Current Long (non-negative) and
            Short (non-positive) Positions of 'assets' in Shares
        price : np.array, pd.Series, list
            Current Price of 'assets' in Dollars
        factor_value : np.array, pd.Series, list
            Factor values of 'assets'
        sector_id : np.array, pd.Series, list
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
        self.portfolio_df['position_value'] = (self.portfolio_df.position *
                                               self.portfolio_df.price)
        self.portfolio_df.sort_values(by=['position_value'],
                                      axis=0, inplace=True,
                                      ascending=False)

    @classmethod
    def from_dataframe(cls, dataframe):
        """Alternative constructor from an asset indexed pd.DataFrame()"""
        asset = dataframe.index
        position = dataframe.iloc[:, 0]
        price = dataframe.iloc[:, 1]
        factor_value = dataframe.iloc[:, 2]
        sector_id = dataframe.iloc[:, 3]
        return cls(asset, position, price, factor_value, sector_id)

    def __repr__(self):
        return repr(self.portfolio_df)

    def __str__(self):
        string = (f"Portfolio: {len(self.portfolio_df.index)} " +
                  f"Assets, ${round(self.long_exposure(),2)} Long, " +
                  f"${round(self.short_exposure(),2)} Short")
        return string

    def __len__(self):
        return len(self.portfolio_df.index)

    def __getitem__(self, position):
        return self.portfolio_df.index[position]

    def longs(self):
        """Returns positions with positive dollar value."""
        return self.portfolio_df[self.portfolio_df.position > 0]

    def shorts(self):
        """Returns positions with negative dollar value."""
        return self.portfolio_df[self.portfolio_df.position < 0]

    def long_exposure(self):
        """Dollar value of longs"""
        longs = self.longs()
        return (longs.position_value).sum()

    def short_exposure(self):
        """Dollar value of shorts"""
        shorts = self.shorts()
        return -(shorts.position_value).sum()

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
        position_value = self.portfolio_df[['sector_id', 'position_value']]
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
                self.portfolio_df.loc[asset, 'position_value'] = (price *
                                        self.portfolio_df.position[asset])
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

            new_position['position_value'] = (new_position.position *
                                              new_position.price)
            self.portfolio_df = self.portfolio_df.append(new_position,
                                                         verify_integrity=True)
        self.portfolio_df.sort_values(by=['position_value'],
                                      axis=0, inplace=True,
                                      ascending=False)
