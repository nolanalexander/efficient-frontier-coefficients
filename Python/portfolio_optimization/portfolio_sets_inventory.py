import pandas as pd

from portfolio_optimization.common_portfolios import process_common_portfolios 
from portfolio_optimization.mean_estimation import mean
from portfolio_optimization.cov_estimation import cov, cov_shrinkage, mgarch_forecast, mgarch_shrinkage

'''
Defines common portfolio sets and settings (mean estimator, cov estimator, etc.)
'''

portfolio_set_settings_df = pd.DataFrame({'standard'         : [mean, cov             , {}, {}, ''                       , '.'               ], 
                                          'cov_shrinkage'    : [mean, cov_shrinkage   , {}, {}, 'cov_shrinkage'          , 'Cov_Shrinkage'   ],
                                          'mgarch'           : [mean, mgarch_forecast , {}, {}, 'mgarch'                 , 'MGARCH'          ],
                                          'mgarch_shrinkage' : [mean, mgarch_shrinkage, {}, {}, 'mgarch_cov_shrinkage'   , 'MGARCH_Shrinkage'], 
                                         }, index=['mean_estimator', 'cov_estimator', 'mean_est_kwargs', 'cov_est_kwargs', 'port_name_ext', 'save_folder']).T
 
def process_common_portfolio_set(portfolio_set_name, assets_set, subfolder_name, start_date, test_start_date, lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='scaling',
                                 run_rolling_window=True, run_yearly_quarterly_monthly=True, find_weights=True, find_coefs=True, transaction_cost=0.01, run_alt_rebalance=False):
    process_common_portfolios(assets_set, subfolder_name, start_date, test_start_date,
                              lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                              run_rolling_window=run_rolling_window, run_yearly_quarterly_monthly=run_yearly_quarterly_monthly, find_weights=find_weights, find_coefs=find_coefs,
                              transaction_cost=transaction_cost, run_alt_rebalance=run_alt_rebalance, **portfolio_set_settings_df.loc[portfolio_set_name])
    
def process_all_common_portfolio_sets(assets_set, subfolder_name, start_date, test_start_date, lower_bound=None, upper_bound=None,max_leverage=1.5, max_leverage_method='constraint',
                                      run_rolling_window=True, run_yearly_quarterly_monthly=True, find_weights=True, find_coefs=True, transaction_cost=0.01, run_alt_rebalance=False):
    for portfolio_set_name in portfolio_set_settings_df.index.values:
        process_common_portfolio_set(portfolio_set_name, assets_set, subfolder_name, start_date, test_start_date,
                                     lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=1.5, max_leverage_method=max_leverage_method,
                                     run_rolling_window=run_rolling_window, run_yearly_quarterly_monthly=run_yearly_quarterly_monthly, find_weights=find_weights, find_coefs=find_coefs, 
                                     transaction_cost=transaction_cost, run_alt_rebalance=run_alt_rebalance)