import unittest
import pandas as pd
import numpy as np
from src.portfolio import Portfolio
from src.construction import dollar_neutral_top_selection


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        self.DEBUG = True
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

    def test_sector_net_exposures(self):
        sector_exp = self.pf.sector_net_exposures()
        self.assertEqual(sector_exp.iloc[0][0], -10)
        self.assertEqual(sector_exp.iloc[1][0], 20)

    def test_dollar_neutral_top_selection(self):

        def trade_portfolio(buy_low=True):
            if buy_low:
                # liquidate not wanted longs
                for stock in pf1.longs().index:
                    if signal[stock] >= 0:
                        pf1.trade(asset=stock, amount=-pf1.position(stock),
                                  price=1, factor_value=signal[stock],
                                  sector_id=1)

                # liquidate not wanted shorts
                for stock in pf1.shorts().index:
                    if signal[stock] <= 0:
                        pf1.trade(asset=stock, amount=-pf1.position(stock),
                                  price=1, factor_value=signal[stock],
                                  sector_id=1)

                # order new longs and shorts
                for stock in new_pos['new_longs']:
                    pf1.trade(asset=stock, amount=10-pf1.position(stock),
                              price=1, factor_value=signal[stock], sector_id=1)

                for stock in new_pos['new_shorts']:
                    pf1.trade(asset=stock, amount=-10-pf1.position(stock),
                              price=1, factor_value=signal[stock], sector_id=1)
            else:
                # liquidate not wanted longs
                for stock in pf1.longs().index:
                    if signal[stock] <= 0:
                        pf1.trade(asset=stock, amount=-pf1.position(stock),
                                  price=1, factor_value=signal[stock],
                                  sector_id=1)

                # liquidate not wanted shorts
                for stock in pf1.shorts().index:
                    if signal[stock] >= 0:
                        pf1.trade(asset=stock, amount=-pf1.position(stock),
                                  price=1, factor_value=signal[stock],
                                  sector_id=1)

                # order new longs and shorts
                for stock in new_pos['new_longs']:
                    pf1.trade(asset=stock, amount=10-pf1.position(stock),
                              price=1, factor_value=signal[stock], sector_id=1)

                for stock in new_pos['new_shorts']:
                    pf1.trade(asset=stock, amount=-10-pf1.position(stock),
                              price=1, factor_value=signal[stock], sector_id=1)

        signal = pd.Series(data={'a': -2, 'b': 0, 'c': 1, 'd': 2, 'e': 3,
                                 'f': -6, 'g': 9, 'h': -10, 'i': 10})
        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1e-8,
                                               short_quantile=1e-8,
                                               min_th=1e-8,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['h'])
        self.assertEqual(new_pos['new_shorts'], ['i'])
        self.assertEqual(new_pos['long_th'], signal['h'])
        self.assertEqual(new_pos['short_th'], signal['i'])
        self.assertEqual(new_pos['current_longs_incr'], [])
        self.assertEqual(new_pos['current_shorts_incr'], [])

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio()

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['h'])
        self.assertEqual(pf1.shorts().index.tolist(), ['i'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=1e-8,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['h', 'f', 'a'])
        self.assertEqual(new_pos['new_shorts'], ['i', 'g', 'e'])
        self.assertEqual(new_pos['long_th'], -1e-8)
        self.assertEqual(new_pos['short_th'], 1e-8)
        self.assertEqual(new_pos['current_longs_incr'], [])
        self.assertEqual(new_pos['current_shorts_incr'], [])

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio()

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['h', 'f', 'a'])
        self.assertEqual(pf1.shorts().index.tolist(), ['i', 'g', 'e'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1e-8,
                                               short_quantile=1e-8,
                                               min_th=1e-8,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['h'])
        self.assertEqual(new_pos['long_th'], 10)
        self.assertEqual(new_pos['short_th'], -10)
        self.assertEqual(new_pos['current_longs_incr'], [])
        self.assertEqual(new_pos['current_shorts_incr'], [])

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a', 'h'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=1e-8,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['h', 'f'])
        self.assertEqual(new_pos['long_th'], 1e-8)
        self.assertEqual(new_pos['short_th'], -1e-8)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a', 'h', 'f'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=0.5,
                                               short_quantile=0.5,
                                               min_th=1e-8,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], ['h', 'f', 'a'])
        self.assertEqual(new_pos['new_shorts'], ['i', 'g', 'e'])
        self.assertEqual(new_pos['long_th'], -1e-8)
        self.assertEqual(new_pos['short_th'], 1)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio()

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['h', 'f', 'a'])
        self.assertEqual(pf1.shorts().index.tolist(), ['i', 'g', 'e'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=0.5,
                                               short_quantile=0.5,
                                               min_th=1e-8,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['h', 'f'])
        self.assertEqual(new_pos['long_th'], 1)
        self.assertEqual(new_pos['short_th'], -1e-8)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a', 'h', 'f'])

        signal = pd.Series(data={'b': 0, 'c': 1, 'd': 2, 'e': 3, 'a': -20,
                                 'f': -6, 'g': -9, 'h': -10, 'i': -10})
        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1e-8,
                                               short_quantile=1e-8,
                                               min_th=1e-8,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], 3)
        self.assertEqual(new_pos['short_th'], -20)
        self.assertEqual(new_pos['current_longs_incr'], ['e'])
        self.assertEqual(new_pos['current_shorts_incr'], ['a'])

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a'])

        signal = pd.Series(data={'b': 0, 'c': 1, 'd': 2, 'e': 3,
                                 'a': -2, 'f': -6, 'g': -9,
                                 'h': -10, 'i': -10})
        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=1e-8,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['i', 'h'])
        self.assertEqual(new_pos['long_th'], 1e-8)
        self.assertEqual(new_pos['short_th'], -1e-8)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a', 'i', 'h'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=11,
                                               buy_low=False)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], 11)
        self.assertEqual(new_pos['short_th'], -11)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=False)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), ['c', 'd', 'e'])
        self.assertEqual(pf1.shorts().index.tolist(), ['a'])

        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=11,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], [])
        self.assertEqual(new_pos['long_th'], -11)
        self.assertEqual(new_pos['short_th'], 11)

        pf1 = Portfolio.from_dataframe(dataframe=self.portfolio_df)

        trade_portfolio(buy_low=True)

        if self.DEBUG:
            print(pf1.longs(), '\n', pf1.shorts())

        self.assertEqual(pf1.longs().index.tolist(), [])
        self.assertEqual(pf1.shorts().index.tolist(), [])

        signal = pd.Series(data={'a': -2, 'f': 2, 'g': np.nan, 'h': -9,
                                 'i': -10, 'j': 3})
        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=1e-8,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['j', 'f'])
        self.assertEqual(new_pos['long_th'], -1e-8)
        self.assertEqual(new_pos['short_th'], 1e-8)

        signal = pd.Series(data={'a': np.nan, 'f': np.nan,
                                 'g': np.nan, 'h': np.nan,
                                 'i': 10, 'j': np.nan})
        new_pos = dollar_neutral_top_selection(self.pf, signal,
                                               long_quantile=1,
                                               short_quantile=1,
                                               min_th=1e-8,
                                               buy_low=True)
        self.assertEqual(new_pos['new_longs'], [])
        self.assertEqual(new_pos['new_shorts'], ['i'])
        self.assertEqual(new_pos['long_th'], -1e-8)
        self.assertEqual(new_pos['short_th'], 10)
