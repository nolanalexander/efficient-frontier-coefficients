import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import matplotlib.pyplot as plt

from data_processing.read_in_data import read_in_eq_weight
from portfolio_optimization.portfolio_utils import calc_portfolio_metrics, get_assets_set_abr

'''
Given the market forecasts, go long or short the equal-weighted portfolio
'''

def run_simple_forecast_backtest(assets_set, subfolder_name, time_interval, forecast_model_name, predictand, predictors_name, test_start_date):
    start_time = time.time()
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    mkt_forecast_df = pd.read_csv('../Portfolios/'+assets_set+'/Mkt_Forecast/'+time_interval+'/'+predictors_name+'/'+forecast_model_name+'/Forecasts/'+predictand+'/'+predictand+'_'+forecast_model_name+'_'+predictors_name+'_forecasts.csv', index_col=0, parse_dates=True)
    ew_returns_df = read_in_eq_weight(assets_set, assets_returns_df.index)
    if time_interval[:8] != 'Rolling_':
        ew_returns_df['Month'] = [dt.datetime(date.year, date.month, 1) for date in ew_returns_df.index]
        simple_rets = (2 * mkt_forecast_df[['Forecast']]-1).shift(1).merge(ew_returns_df[['Month', 'Chg']], left_index=True, right_on='Month').drop(columns=['Month']).dropna().prod(1)
    else:
        simple_rets = (2 * mkt_forecast_df[['Forecast']]-1).shift(1).merge(ew_returns_df[['Chg']], left_index=True, right_index=True).dropna().prod(1)
    simple_port_vals = (1+simple_rets).cumprod()
    
    # Set up directories
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    mf_plot_dir = subfolder_dir+'Plots/Mkt_Forecast/'+time_interval+'/'+forecast_model_name+'/'
    if not os.path.exists(mf_plot_dir):
        os.makedirs(mf_plot_dir)
    mf_metrics_dir = subfolder_dir+'Metrics/Mkt_Forecast/'+time_interval+'/'+forecast_model_name+'/'
    if not os.path.exists(mf_metrics_dir):
        os.makedirs(mf_metrics_dir)
    
    port_vals_df = pd.read_csv(subfolder_dir+'Portfolio_Vals/portfolio_vals_'+str(test_start_date.year)+'to'+str(assets_returns_df.index.year[-1])+'.csv', index_col=0, parse_dates=True)
    port_vals_df = port_vals_df[['SPX', 'Eq_Weights']].copy()
    
    # Plot Portfolios
    assets_set_abr = get_assets_set_abr(assets_set)
    plt.figure(figsize=(16,8))
    plt.plot(simple_port_vals.index, np.log(simple_port_vals), label='Simple Forecast Backtest')
    plt.plot(port_vals_df['SPX'].index, np.log(port_vals_df['SPX']), label='SPX')
    plt.plot(port_vals_df['Eq_Weights'].index, np.log(port_vals_df['Eq_Weights']), label='Equal Weights')
    plt.title(assets_set_abr + 'Binary Market Forecast Simple Backtest')
    plt.xlabel('Year')
    plt.ylabel('Log Return')
    plt.legend(loc='upper left')
    start_end_name = str(test_start_date.year)+'to'+str(assets_returns_df.index.max().year)
    plt.savefig(mf_plot_dir+'mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_simple_backtest_'+time_interval.lower()+'_'+start_end_name+'.png', dpi=500)
    plt.close()
    
    # Portfolio Metrics
    port_metrics_df = pd.DataFrame({'Mkt_Forecast_Simple' : calc_portfolio_metrics(simple_port_vals)}).T
    port_metrics_df.to_csv(mf_metrics_dir+'mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_simple_metrics_'+time_interval.lower()+'_'+start_end_name+'.csv')
    print('Simple Forecast Backtest Runtime:', str(round((time.time() - start_time)/(60))), 'mins')
    