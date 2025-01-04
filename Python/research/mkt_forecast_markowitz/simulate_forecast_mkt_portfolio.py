import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from data_processing.read_in_data import read_in_rf
from portfolio_optimization.simulate_portfolios import get_portfolio_vals 
from portfolio_optimization.portfolio_utils import calc_portfolio_metrics, alpha_regression_ttest_baseline_port_vals, get_assets_set_abr

'''
Given the daily market forecast implied portfolio weights,
gets the portfolio vals, calculates metrics, and plots
'''

def simulate_forecast_mkt_portfolio(assets_set, subfolder_name, time_interval, forecast_model_name, predictors_name, test_start_date, max_leverage=1.5):
    start_time = time.time()
    # Set up directories
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    mf_weights_dir = subfolder_dir+'Weights/Mkt_Forecast/'+time_interval+'/'
    mf_port_vals_dir = subfolder_dir+'Portfolio_Vals/Mkt_Forecast/'+time_interval+'/'+forecast_model_name+'/'
    if not os.path.exists(mf_port_vals_dir):
        os.makedirs(mf_port_vals_dir)
    mf_plot_dir = subfolder_dir+'Plots/Mkt_Forecast/'+time_interval+'/'+forecast_model_name+'/'
    if not os.path.exists(mf_plot_dir):
        os.makedirs(mf_plot_dir)
    mf_metrics_dir = subfolder_dir+'Metrics/Mkt_Forecast/'+time_interval+'/'+forecast_model_name+'/'
    if not os.path.exists(mf_metrics_dir):
        os.makedirs(mf_metrics_dir)
    
    # Read in data
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    test_assets_returns_df = assets_returns_df[assets_returns_df.index >= test_start_date]
    start_end_name = str(test_start_date.year)+'to'+str(assets_returns_df.index.max().year)
    port_vals_df = pd.read_csv(subfolder_dir+'Portfolio_Vals/portfolio_vals_'+start_end_name+'.csv', index_col=0, parse_dates=True)
    
    # Save Portfolio Vals
    mkt_forecast_weights_df = pd.read_csv(mf_weights_dir+'daily_weights_mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_'+time_interval.lower()+'.csv', index_col=0, parse_dates=True)
    mf_port_vals_df = get_portfolio_vals(mkt_forecast_weights_df, test_assets_returns_df)
    mf_port_vals_df.to_csv(mf_port_vals_dir+'mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_portfolio_vals_'+time_interval.lower()+'_'+start_end_name+'.csv')
    
    # Plot Portfolios
    assets_set_abr = get_assets_set_abr(assets_set)
    plt.figure(figsize=(16,8))
    plt.plot(mf_port_vals_df.index, np.log(mf_port_vals_df['Port_Vals']), label='Mkt Forecast Tangency Portfolio')
    plt.plot(port_vals_df['SPX'].index, np.log(port_vals_df['SPX']), label='SPX')
    plt.plot(port_vals_df['Eq_Weights'].index, np.log(port_vals_df['Eq_Weights']), label='Equal Weights')
    baseline_name = 'Markowitz_Tan_Port_'+(time_interval.lower()[-3:] if time_interval.lower()[:7] == 'rolling' else time_interval.lower())
    plt.plot(port_vals_df.index, np.log(port_vals_df[baseline_name]), label='Markowitz Tangency Portfolio')
    plt.title(assets_set_abr + ' EF Coefs CART Market Forecast Tangency Portfolio to Benchmarks')
    plt.xlabel('Year')
    plt.ylabel('Log Return')
    plt.legend(loc='upper left')
    plt.savefig(mf_plot_dir+'mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_tan_port_vals_'+time_interval.lower()+'_'+start_end_name+'.png')
    plt.close()
    
    # Portfolio Metrics
    port_metrics_df = pd.DataFrame({'Mkt_Forecast_Tan_Port' : calc_portfolio_metrics(mf_port_vals_df['Port_Vals'])}).T
    port_metrics_df.to_csv(mf_metrics_dir+'mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_tan_port_metrics_'+time_interval.lower()+'_'+start_end_name+'.csv')
    
    # Alpha Regression ttests
    port_vals_df = pd.read_csv(subfolder_dir+'Portfolio_Vals/portfolio_vals_'+start_end_name+'.csv', index_col=0, parse_dates=True)
    port_vals_df = port_vals_df[[baseline_name, 'SPX', 'Eq_Weights']].copy()
    rfs = read_in_rf(port_vals_df.index)
    alpha_pval_df = alpha_regression_ttest_baseline_port_vals(port_vals_df, mf_port_vals_df['Port_Vals'], rfs)
    alpha_pval_df.to_csv(mf_metrics_dir + 'markowitz_mkt_forecast_alpha_ttest.csv')
    print('Simulate Forecast Market Portfolio Runtime:', str(round((time.time() - start_time)/(60))), 'mins')
    