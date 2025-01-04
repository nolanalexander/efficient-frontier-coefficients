import numpy as np
import pandas as pd

from portfolio_optimization.black_litterman import calc_black_litterman_tan_port_weights, find_market_equilibrium_weights
from data_processing.read_in_data import read_in_rf


assets_set = 'SPDR_ETF_Sectors'

assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
assets_returns_df = assets_returns_df.iloc[-252:]
rfs = read_in_rf(assets_returns_df.index)
portfolio_views = pd.DataFrame([[0,   0,   0, 0.5, 0,   0, -0.5, 0, 0, 0],
                                [0,   0.6, 0, 0,  -0.2, 0,  0.6, 0, 0, 0],
                                [0.1, 0,   0, 0.2, 0,   0, -0.3, 0, 0, 0],
                                [0,   0,   0, 0,   0,   1,  0  , 0, 0, 0]],
                               columns=assets_returns_df.columns)
expected_returns_each_view = pd.Series([0.05 , 0.1 , 0.05 , 0.1]).T
weight_on_views = 0.01
# market_equilibrium_weights = find_market_equilibrium_weights(assets_set, assets_returns_df.columns.to_list(), assets_returns_df.index[-1])
market_equilibrium_weights = pd.Series(np.repeat(1/len(assets_returns_df.columns), len(assets_returns_df.columns)), index=assets_returns_df.columns)

bl_tan_weights = calc_black_litterman_tan_port_weights(assets_returns_df, rfs.iloc[-1], portfolio_views, expected_returns_each_view, 
                                                       weight_on_views, market_equilibrium_weights, levels_of_unconfidence=None,
                                                       reweight_P=False, lower_bound=None, upper_bound=None, 
                                                       max_leverage=1.5, max_leverage_method='constraint')