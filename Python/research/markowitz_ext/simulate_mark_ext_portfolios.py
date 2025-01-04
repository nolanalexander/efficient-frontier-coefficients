import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from data_processing.read_in_data import read_in_rf
from portfolio_optimization.portfolio_utils import calc_portfolio_metrics, alpha_regression_ttest_baseline_port_vals, get_assets_set_abr
from portfolio_optimization.simulate_portfolios import simulate_portfolio

'''
Given the daily Markowitz extension portfolio weights,
gets the portfolio vals, calculates metrics, and plots
'''

def simulate_mark_ext_portfolios(assets_set, subfolder_name, lookback_method, test_start_date, transaction_cost=0.01, leverage=1):
    start_time = time.time()
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    me_weights_dir = subfolder_dir+'Weights/Markowitz_Ext/'
    if not os.path.exists(me_weights_dir):
        os.makedirs(me_weights_dir)
    me_port_vals_dir = subfolder_dir+'Portfolio_Vals/Markowitz_Ext/'
    if not os.path.exists(me_port_vals_dir):
        os.makedirs(me_port_vals_dir)
        
    mark_ext_weights = pd.read_csv(me_weights_dir+'daily_weights_mark_ext_'+lookback_method+'.csv', index_col=0, parse_dates=True)
    mark_ext_weights_filename = 'daily_weights_mark_ext_'+lookback_method+'.csv'
    
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    
    if(test_start_date is not None):
        mark_ext_weights = mark_ext_weights[mark_ext_weights.index >= test_start_date]
        assets_returns_df = assets_returns_df[assets_returns_df.index >= test_start_date]
        test_start_year = test_start_date.year
    else:
        test_start_year = mark_ext_weights.index[0].year
    
    ext_port_vals_df = simulate_portfolio(assets_set, subfolder_name, mark_ext_weights_filename, test_start_date, transaction_cost=transaction_cost, 
                                          save_folder='Markowitz_Ext')
    ext_port_vals_df = (1 + ext_port_vals_df.pct_change() * leverage).cumprod()
    
    # Save portfolio vals over time
    port_vals_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Portfolio_Vals/'
    if not os.path.exists(port_vals_dir):
        os.makedirs(port_vals_dir)
    ext_port_vals_df.to_csv(me_port_vals_dir+f'mark_ext_portfolio_vals_{test_start_year}to{ext_port_vals_df.index.year[-1]}.csv', index=True)
    
    plot_mark_ext_portfolios(assets_set, subfolder_name, ext_port_vals_df, test_start_date, leverage=leverage)
    calc_mark_ext_portfolio_metrics(assets_set, subfolder_name, ext_port_vals_df.iloc[1:], test_start_date)
    
    print('Simulated Mark Ext Portfolios Runtime:', round((time.time() - start_time)/60),' mins' )
    return ext_port_vals_df
    
def plot_mark_ext_portfolios(assets_set, subfolder_name, port_vals_df, test_start_date, leverage=1):
    assets_set_abr = get_assets_set_abr(assets_set)
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    me_port_vals_dir = subfolder_dir+'Portfolio_Vals/Markowitz_Ext/'
    me_plot_dir = subfolder_dir+'Plots/Markowitz_Ext/'   
    if not os.path.exists(me_plot_dir):
        os.makedirs(me_plot_dir)
    end_year = port_vals_df.index.year[-1]
    port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
    mark_ext_port_vals_df = pd.read_csv(me_port_vals_dir+f'mark_ext_portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
    
    # Plot portfolio returns relative to baselines
    plt.figure(figsize=(10, 5))
    leverage_name = f' {leverage}x Levered' if leverage > 1 else ''
    plt.plot(port_vals_df.index, np.log(port_vals_df['Eq_Weights']), label='Equal Weighted')
    plt.plot(port_vals_df.index, np.log(port_vals_df['SPX']), label='S&P 500')
    plt.plot(port_vals_df.index, np.log(port_vals_df['60/40']), label='60/40 Stocks and Bonds')
    plt.plot(port_vals_df.index, np.log(port_vals_df['Markowitz_Tan_Port_1mo']), label='Tangency Portfolio rolling 1mo')
    plt.plot(port_vals_df.index, np.log(mark_ext_port_vals_df['Port_Vals']), label='Min Distance Portfolio to Tangency rolling 1mo'+leverage_name, c='black')
    title = assets_set_abr + ' Min Distance Portfolio to Benchmarks'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Log Portfolio Value')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left')
    end_year = port_vals_df.index.year[-1]
    plot_filename = f'Markowitz_Ext_Portfolio_Returns_to_Baseline_{test_start_date.year}to{end_year}.png'
    plt.savefig(me_plot_dir+'/'+plot_filename)
    plt.close()

def calc_mark_ext_portfolio_metrics(assets_set, subfolder_name, me_port_vals_df, start_date):
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    me_metrics_dir = subfolder_dir+'Metrics/Markowitz_Ext/'
    if not os.path.exists(me_metrics_dir):
        os.makedirs(me_metrics_dir) 
    end_year = me_port_vals_df.index.year[-1]
    
    # Standard metrics
    port_metrics_df = pd.DataFrame({'Markowitz_Ext' : calc_portfolio_metrics(me_port_vals_df['Port_Vals'])})
    metrics_filename = f'portfolio_metrics_{start_date.year}to{end_year}.csv'
    port_metrics_df.to_csv(me_metrics_dir+metrics_filename)
    
    # Alpha Regression ttests
    port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
    port_vals_df = port_vals_df[['Markowitz_Tan_Port_1mo', 'SPX', 'Eq_Weights', '60/40']]
    rfs = read_in_rf(port_vals_df.index)
    alpha_pval_df = alpha_regression_ttest_baseline_port_vals(port_vals_df, me_port_vals_df['Port_Vals'], rfs)
    alpha_pval_df.to_csv(me_metrics_dir + 'markowitz_ext_alpha_ttest.csv')
    return port_metrics_df
    
