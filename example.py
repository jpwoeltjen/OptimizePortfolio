import pandas as pd
from src.op import Portfolio

assets = ['a', 'b', 'c', 'd', 'e']
position = [-10, -5, 3, 2, 1]
price = [1, 2, 10/3, 5, 10]
factor_value = [-1, 0, 1, 2, 3]
sector_id = [1, 1, 2, 2, 1]

portfolio_df = pd.DataFrame(index=assets, data={'position': position,
                                                'price': price,
                                                'factor_value': factor_value,
                                                'sector_id': sector_id})

pf = Portfolio.from_dataframe(dataframe=portfolio_df)
print(pf)
print(pf.sector_net_exposures())
pf.trade(asset='a', amount=10, price=1.1)
print(pf)
print(pf.sector_net_exposures())


print('longs', pf.longs())
print('shorts', pf.shorts())

signal = pd.Series(data={'a': -2, 'f': -6, 'g': 9, 'h': -10, 'i': 10})
new_pos = pf.dollar_neutral_top_selection(signal, long_quantile=1,
                                          short_quantile=1, min_th=0,
                                          buy_low=True)
print(new_pos)

pf.trade(asset=new_pos['new_shorts'][1],
         amount=-10, price=1.1, sector_id=1)
print(pf)
print('longs', pf.longs())
print('shorts', pf.shorts())
print('sector net exposure:\n', pf.sector_net_exposures())
print('len', len(pf))
print('first=largest:\n', pf[0])
