import unittest
import pandas as pd
import numpy as np
from src.OptimizePortfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        assets = ['a', 'b', 'c', 'd', 'e']
        position = [-10, -5, 3, 2, 1]
        price = [1, 2, 10/3, 5, 10]
        factor_value = [-1, 0, 1, 2, 3]
        sector_id = [1, 1, 2, 2, 1]

        self.portfolio_df = pd.DataFrame(index=assets,
                                         data={'position': position,
                                               'price': price,
                                               'factor_value': factor_value,
                                               'sector_id': sector_id})

        self.pf = Portfolio.from_dataframe(dataframe=self.portfolio_df)

    def test_longs(self):
        long_stocks = list(self.pf.longs().index)
        long_stocks_truth = list(
          self.portfolio_df[self.portfolio_df.position > 0].index)
        self.assertEqual(long_stocks, long_stocks_truth)

    def test_shorts(self):
        short_stocks = list(self.pf.shorts().index)
        short_stocks_truth = list(
          self.portfolio_df[self.portfolio_df.position < 0].index)
        self.assertEqual(short_stocks, short_stocks_truth)

    def test_dollar_neutral_top_selection(self):
        signal = pd.Series(data={'a': -2, 'f': -6, 'g': 9, 'h': -10, 'i': 10})
        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1e-8,
                                                       short_quantile=1e-8,
                                                       min_th=0,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['i'])
        self.assertEqual(new_pos['long_th'], signal['h'])
        self.assertEqual(new_pos['short_th'], signal['i'])

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=0,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['h'])
        self.assertEqual(new_pos['new_shorts'], ['i', 'g'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], 0)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1e-8,
                                                       short_quantile=1e-8,
                                                       min_th=0,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['h'])
        self.assertEqual(new_pos['long_th'], 10)
        self.assertEqual(new_pos['short_th'], -10)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=0,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], ['i'])
        self.assertEqual(new_pos['new_shorts'], ['h', 'f'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], 0)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=0.5,
                                                       short_quantile=0.5,
                                                       min_th=0,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['h'])
        self.assertEqual(new_pos['new_shorts'], ['i', 'g'])
        self.assertEqual(new_pos['long_th'], -2)
        self.assertEqual(new_pos['short_th'], 0)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=0.5,
                                                       short_quantile=0.5,
                                                       min_th=0,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], ['i'])
        self.assertEqual(new_pos['new_shorts'], ['h', 'f'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], -2)

        signal = pd.Series(data={'a': -20, 'f': -6, 'g': -9,
                                 'h': -10, 'i': -10})
        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1e-8,
                                                       short_quantile=1e-8,
                                                       min_th=0,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], -20)

        signal = pd.Series(data={'a': -2, 'f': -6, 'g': -9,
                                 'h': -10, 'i': -10})
        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=0,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['i'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], -2)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=11,
                                                       buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], 11)
        self.assertEqual(new_pos['short_th'], -11)

        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=11,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], -11)
        self.assertEqual(new_pos['short_th'], 11)

        signal = pd.Series(data={'a': -2, 'f': 2, 'g': np.nan, 'h': -9,
                                 'i': -10, 'j': 3})
        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=0,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['i'])
        self.assertEqual(new_pos['new_shorts'], ['j', 'f'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], 0)

        signal = pd.Series(data={'a': np.nan, 'f': np.nan,
                                 'g': np.nan, 'h': np.nan,
                                 'i': 10, 'j': np.nan})
        new_pos = self.pf.dollar_neutral_top_selection(signal,
                                                       long_quantile=1,
                                                       short_quantile=1,
                                                       min_th=0,
                                                       buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['i'])
        self.assertEqual(new_pos['long_th'], 0)
        self.assertEqual(new_pos['short_th'], 10)

    def test_sector_net_exposures(self):
        sector_exp = self.pf.sector_net_exposures()
        self.assertEqual(sector_exp.iloc[0][0], -10)
        self.assertEqual(sector_exp.iloc[1][0], 20)
