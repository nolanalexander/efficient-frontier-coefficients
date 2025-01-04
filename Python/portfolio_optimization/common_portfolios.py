import pandas as pd
import numpy as np
import time
import os

from data_processing.read_in_data import read_in_rf
from portfolio_optimization.markowitz import calc_all_markowitz_weights_and_coefs
from portfolio_optimization.simulate_portfolios import simulate_portfolio, plot_portfolios
from portfolio_optimization.portfolio_utils import calc_portfolio_metrics
from portfolio_optimization.mean_estimation import mean
from portfolio_optimization.cov_estimation import cov

'''
Processes common portfolios: tangency with different lookbacks, and MVP.
'''

def calc_all_common_markowitz_weights_and_coefs(assets_set, subfolder_name, port_name_ext='', 
                                                mean_estimator=mean, cov_estimator=cov, mean_est_args=[], cov_est_args=[],
                                                lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, 
                                                run_rolling_window=True, run_yearly_quarterly_monthly=True, find_weights=True, find_coefs=True, save_folder='.'):
    start_time = time.time()
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    rfs = read_in_rf(assets_returns_df.index)

    if run_rolling_window:
        calc_all_markowitz_weights_and_coefs('tan_rolling_1yr'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=252,
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound,max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_6mo'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/2),
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound,max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_3mo'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/4),
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound,max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_1mo'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/12),
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_1mo_ret_3mo_cov'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/12), cov_lookback=int(252/4),
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_1mo_ret_1yr_cov'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs,
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/12), cov_lookback=252,
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_rolling_3mo_ret_1yr_cov'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs,
                                              portfolio_selection='tangency', lookback_method='rolling', lookback=int(252/4), cov_lookback=252,
                                              mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                              lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                              find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('mvp_rolling_3mo'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                             portfolio_selection='MVP', lookback_method='rolling', lookback=int(252/4),
                                             mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                             lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                             find_weights=find_weights, find_coefs=False, save_folder=save_folder)
    if run_yearly_quarterly_monthly:
        calc_all_markowitz_weights_and_coefs('tan_yearly'+port_name_ext, assets_set, subfolder_name, assets_returns_df, rfs, 
                                             portfolio_selection='tangency', lookback_method='yearly', 
                                             mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                             lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                             find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_quarterly'+port_name_ext, assets_set,subfolder_name, assets_returns_df, rfs, 
                                             portfolio_selection='tangency', lookback_method='quarterly', 
                                             mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                             lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                             find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
        calc_all_markowitz_weights_and_coefs('tan_monthly'+port_name_ext, assets_set,subfolder_name, assets_returns_df, rfs, 
                                             portfolio_selection='tangency', lookback_method='monthly', 
                                             mean_estimator=mean_estimator, cov_estimator=cov_estimator,
                                             lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                             find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
    run_time = round((time.time() - start_time)/60, 1)
    time_interval = 'mins' if run_time < 60 else 'hrs'
    run_time = run_time if run_time < 60 else round(run_time/60, 1)
    print('Markowitz Optimization Runtime:', run_time, time_interval)


def simulate_all_common_portfolios(assets_set, subfolder_name, start_date, transaction_cost=0.01, port_name_ext='', save_folder='.', run_alt_rebalance=False):
    weights_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Weights/'
    
    mark_tan_port_weights_3mo = pd.read_csv(weights_dir+'daily_weights_tan_rolling_3mo.csv', index_col=0, parse_dates=True)
    start_year = start_date.year if start_date is not None else mark_tan_port_weights_3mo.index[0].year
    end_year = mark_tan_port_weights_3mo.index[-1].year
    
    port_settings_df = pd.DataFrame({
        'Markowitz_Tan_Port_1yr'                 : ['daily_weights_tan_rolling_1yr'+port_name_ext+'.csv'            , 1   , None, 0],
        'Markowitz_Tan_Port_6mo'                 : ['daily_weights_tan_rolling_6mo'+port_name_ext+'.csv'            , 1   , None, 0],
        'Markowitz_Tan_Port_1mo'                 : ['daily_weights_tan_rolling_1mo'+port_name_ext+'.csv'            , 1   , None, 0],
        'Markowitz_Tan_Port_1mo_ret_3mo_cov'     : ['daily_weights_tan_rolling_1mo_ret_3mo_cov'+port_name_ext+'.csv', 1   , None, 0],
        'Markowitz_Tan_Port_1mo_ret_1yr_cov'     : ['daily_weights_tan_rolling_1mo_ret_1yr_cov'+port_name_ext+'.csv', 1   , None, 0],
        'Markowitz_Tan_Port_3mo_ret_1yr_cov'     : ['daily_weights_tan_rolling_3mo_ret_1yr_cov'+port_name_ext+'.csv', 1   , None, 0],
        'Markowitz_Tan_Port_yearly'              : ['daily_weights_tan_yearly'+port_name_ext+'.csv'                 , 1   , None, 0],
        'Markowitz_Tan_Port_quarterly'           : ['daily_weights_tan_quarterly'+port_name_ext+'.csv'              , 1   , None, 0],
        'Markowitz_Tan_Port_monthly'             : ['daily_weights_tan_monthly'+port_name_ext+'.csv'                , 1   , None, 0],
        'Markowitz_Tan_Port_3mo'                 : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 1   , None, 0],
        'Markowitz_MVP_3mo'                      : ['daily_weights_mvp_rolling_3mo'+port_name_ext+'.csv'            , 1   , None, 0],
        'Markowitz_Tan_Port_3mo_rebalance_1wk'   : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 5   , None, 1],
        'Markowitz_Tan_Port_3mo_rebalance_2wk'   : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 10  , None, 1],
        'Markowitz_Tan_Port_3mo_rebalance_1mo'   : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 21  , None, 1],
        'Markowitz_Tan_Port_3mo_rebalance_3mo'   : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 63  , None, 1],
        'Markowitz_Tan_Port_3mo_rebalance_1yr'   : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , 252 , None, 1],
        'Markowitz_Tan_Port_3mo_rebalance_5perc' : ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , None, 0.05, 1],
        'Markowitz_Tan_Port_3mo_rebalance_10perc': ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , None, 0.10, 1],
        'Markowitz_Tan_Port_3mo_rebalance_15perc': ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , None, 0.15, 1],
        'Markowitz_Tan_Port_3mo_rebalance_20perc': ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , None, 0.20, 1],
        'Markowitz_Tan_Port_3mo_rebalance_25perc': ['daily_weights_tan_rolling_3mo'+port_name_ext+'.csv'            , None, 0.25, 1],
        'Markowitz_Perf_Tan_3mo'                 : ['perfect_tan_3mo'                                               , None, 0.25, 0],
        'Eq_Weights'                             : ['eq_weights'                                                    , 1   , None, 0],
        'SPX'                                    : ['spx'                                                           , 1   , None, 0],
        'Bonds'                                  : ['bonds'                                                         , 1   , None, 0],
        '60/40'                                  : ['60/40'                                                         , 1   , None, 0],
        }, index=['Port_Filename', 'Update_Freq', 'Abs_Dev_Update', 'Is_Alt_Rebalance']).T
    
    full_port_vals_df = pd.DataFrame()
    for portfolio_name in port_settings_df.index:
        weights_filename, update_freq, abs_dev_update, port_is_alt_rebalance = port_settings_df.loc[portfolio_name]
        if not port_is_alt_rebalance or run_alt_rebalance:
            full_port_vals_df = pd.concat((full_port_vals_df, 
                                           simulate_portfolio(assets_set, subfolder_name, weights_filename, start_date, 
                                                              transaction_cost=transaction_cost, update_freq=update_freq, 
                                                              abs_dev_update=abs_dev_update, portfolio_name=portfolio_name, 
                                                              port_name_ext=port_name_ext, save_folder=save_folder)))
    def decompose_tensor_to_df(tensor_df, indices, column):
        df = tensor_df.set_index(indices)[[column]].unstack(level=-1).copy()
        df.columns = df.columns.levels[1].values
        return df
    port_vals_df = decompose_tensor_to_df(full_port_vals_df, ['Date', 'Portfolio'], 'Port_Vals')
    port_vals_before_fees_df = decompose_tensor_to_df(full_port_vals_df, ['Date', 'Portfolio'], 'Port_Vals_before_fees')
    fees_over_time_df = decompose_tensor_to_df(full_port_vals_df, ['Date', 'Portfolio'], 'Fees')
    
    # Save portfolio vals over time
    port_vals_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Portfolio_Vals/'+save_folder+'/'
    if not os.path.exists(port_vals_dir):
        os.makedirs(port_vals_dir)
    port_vals_df.to_csv(port_vals_dir+'portfolio_vals'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.csv')
    port_vals_before_fees_df.to_csv(port_vals_dir+'portfolio_vals_before_fees'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.csv')
    fees_over_time_df.to_csv(port_vals_dir+'fee_vals'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.csv')
    return full_port_vals_df
    
def plot_all_common_portfolios(assets_set, subfolder_name, start_date, port_name_ext='', save_folder='.', run_alt_rebalance=False):
    weights_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Weights/'
    mark_tan_port_weights_3mo = pd.read_csv(weights_dir+'daily_weights_tan_rolling_3mo.csv', index_col=0, parse_dates=True)
    start_year = start_date.year if start_date is not None else mark_tan_port_weights_3mo.index[0].year
    end_year = mark_tan_port_weights_3mo.index[-1].year
    
    port_vals_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Portfolio_Vals/'+save_folder+'/'
    port_vals_df = pd.read_csv(port_vals_dir+'portfolio_vals'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.csv', index_col=0, parse_dates=True)
    plot_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Plots/'+save_folder+'/'
    
    # Plot portfolio returns of rolling sensitivity analysis
    label_by_port = {
        'Markowitz_Tan_Port_1yr'             : 'Tangency Portfolio rolling 1yr',
        'Markowitz_Tan_Port_6mo'             : 'Tangency Portfolio rolling 6mo',
        'Markowitz_Tan_Port_3mo'             : 'Tangency Portfolio rolling 3mo',
        'Markowitz_Tan_Port_1mo'             : 'Tangency Portfolio rolling 1mo',
        'Markowitz_Tan_Port_1mo_ret_3mo_cov' : 'Tangency Portfolio rolling 1mo ret 3mo cov',
        'Markowitz_Tan_Port_1mo_ret_1yr_cov' : 'Tangency Portfolio rolling 1mo ret 1yr cov',
        'Markowitz_Tan_Port_3mo_ret_1yr_cov' : 'Tangency Portfolio rolling 3mo ret 1yr cov'
        }
    plot_portfolios(port_vals_df, label_by_port, 'Tangency Portfolio Rolling Lookback Sensitivity Analysis', plot_dir, 
                    'Portfolio_Returns_rolling'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.png', scale='log')
    
    # Plot portfolio returns of yearly, quarterly, and monthly
    label_by_port = {
        'Markowitz_Tan_Port_3mo'       : 'Markowitz Tangency Portfolio rolling 3mo',
        'Markowitz_Tan_Port_yearly'    : 'Markowitz Tangency Portfolio Yearly',
        'Markowitz_Tan_Port_quarterly' : 'Markowitz Tangency Portfolio Quarterly',
        'Markowitz_Tan_Port_monthly'   : 'Markowitz Tangency Portfolio Monthly'
        }
    plot_portfolios(port_vals_df, label_by_port, 'Tangency Portfolio Time Interval Lookback Sensitivity Analysis', plot_dir, 
                    'Portfolio_Returns_rolling_vs_not'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.png', scale='log')

    # Plot tangency portfolio returns of different rebalancing periods
    if run_alt_rebalance:
        label_by_port = {
            'Markowitz_Tan_Port_3mo'                  : 'Rebalance daily',
            'Markowitz_Tan_Port_3mo_rebalance_1wk'    : 'Rebalance weekly',
            'Markowitz_Tan_Port_3mo_rebalance_2wk'    : 'Rebalance every 2 weeks',
            'Markowitz_Tan_Port_3mo_rebalance_1mo'    : 'Rebalance monthly',
            'Markowitz_Tan_Port_3mo_rebalance_3mo'    : 'Rebalance quarterly',
            'Markowitz_Tan_Port_3mo_rebalance_1yr'    : 'Rebalance yearly',
            'Markowitz_Tan_Port_3mo_rebalance_5perc'  : 'Rebalance abs chg > 5%',
            'Markowitz_Tan_Port_3mo_rebalance_10perc' : 'Rebalance abs chg > 10%',
            'Markowitz_Tan_Port_3mo_rebalance_15perc' : 'Rebalance abs chg > 15%',
            'Markowitz_Tan_Port_3mo_rebalance_20perc' : 'Rebalance abs chg > 20%',
            'Markowitz_Tan_Port_3mo_rebalance_25perc' : 'Rebalance abs chg > 25%',
            }
        plot_portfolios(port_vals_df, label_by_port, 'Tangency Portfolio 3mo Rolling Rebalance Sensitivity Analysis', plot_dir,
                        'Portfolio_Returns_3mo_rolling_rebalancing'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.png', scale='log')
    
    # Calculate portfolio metrics
    port_metrics_df = port_vals_df.apply(calc_portfolio_metrics).T
    metrics_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Metrics/'+save_folder+'/'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    metrics_filename = 'portfolio_metrics'+port_name_ext+'_'+str(start_year)+'to'+str(end_year)+'.csv'
    port_metrics_df.to_csv(metrics_dir+metrics_filename)
    
def process_common_portfolios(assets_set, subfolder_name, start_date, test_start_date,
                              lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, 
                              run_rolling_window=True, run_yearly_quarterly_monthly=True, find_weights=True, find_coefs=True,
                              transaction_cost=0.01, 
                              mean_estimator=np.mean, cov_estimator=np.cov, mean_est_kwargs={}, cov_est_kwargs={},
                              port_name_ext='', save_folder='.', run_alt_rebalance=False):
    port_name_ext = port_name_ext if port_name_ext == '' or port_name_ext[0] == '_' else '_'+port_name_ext
    calc_all_common_markowitz_weights_and_coefs(assets_set, subfolder_name, port_name_ext=port_name_ext, 
                                                mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_args=mean_est_kwargs, cov_est_args=cov_est_kwargs,
                                                lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                run_rolling_window=run_rolling_window, run_yearly_quarterly_monthly=run_yearly_quarterly_monthly, 
                                                find_weights=find_weights, find_coefs=find_coefs, save_folder=save_folder)
    start_time = time.time()
    simulate_all_common_portfolios(assets_set, subfolder_name, start_date, transaction_cost=transaction_cost, 
                                    port_name_ext=port_name_ext, save_folder=save_folder, run_alt_rebalance=run_alt_rebalance)
    plot_all_common_portfolios(assets_set, subfolder_name, start_date, port_name_ext=port_name_ext, save_folder=save_folder, run_alt_rebalance=run_alt_rebalance)

    simulate_all_common_portfolios(assets_set, subfolder_name, test_start_date, transaction_cost=transaction_cost, 
                                   port_name_ext=port_name_ext, save_folder=save_folder, run_alt_rebalance=run_alt_rebalance)
    plot_all_common_portfolios(assets_set, subfolder_name, test_start_date, port_name_ext=port_name_ext, save_folder=save_folder, run_alt_rebalance=run_alt_rebalance)
    print('Simulate and Plot Portfolios Runtime:', round((time.time() - start_time)/60), 'mins' )
    
