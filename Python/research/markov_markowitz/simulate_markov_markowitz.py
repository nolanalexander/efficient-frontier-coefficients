import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from data_processing.read_in_data import read_in_rf
from portfolio_optimization.portfolio_utils import calc_portfolio_metrics, alpha_regression_ttest_baseline_port_vals, get_assets_set_abr
from portfolio_optimization.simulate_portfolios import get_portfolio_vals

'''
Simulates the Markov Markowitz model: 
Given the portfolio states to weights mapping, 
calculates the weighted sum of state weights from probabilities of state transitions
'''

def simulate_markov_markowitz(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, test_start_date, in_sample=False):
    start_time = time.time()
    
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    
    include_date = (assets_returns_df.index < test_start_date) if in_sample else (assets_returns_df.index >= test_start_date)
    test_dates = assets_returns_df.index[include_date].to_series()
    test_assets_returns_df = assets_returns_df.loc[test_dates]
    
    subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    end_year = assets_returns_df.index[-1].year
    port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
    tan_port_vals = port_vals_df['Markowitz_Tan_Port_'+(time_interval.lower() if time_interval[:8] != 'Rolling_' else time_interval[8:])]
    spx_port_vals = port_vals_df['SPX']
    eq_weights_port_vals = port_vals_df['Eq_Weights']
    
    weights_dir, plot_dir, port_vals_dir, metrics_dir = ['../Portfolios/'+assets_set+'/Versions/'+subfolder_name+ending_dir for ending_dir in ['/Weights/', '/Plots/', '/Portfolio_Vals/', '/Metrics/']]
    assets_set_abr = get_assets_set_abr(assets_set)
    
    for clustering_model in clustering_models:
        for num_clusters in num_clusters_list:
            # print(clustering_model, num_clusters)
            
            # Set up directories
            mm_subfolder = 'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
            mm_weights_dir = weights_dir + mm_subfolder
            mm_plot_dir = plot_dir + mm_subfolder
            if not os.path.exists(mm_plot_dir):
                os.makedirs(mm_plot_dir)
            mm_port_vals_dir = port_vals_dir + mm_subfolder
            if not os.path.exists(mm_port_vals_dir):
                os.makedirs(mm_port_vals_dir)
            mm_metrics_dir = metrics_dir + mm_subfolder
            if not os.path.exists(mm_metrics_dir):
                os.makedirs(mm_metrics_dir)
                
            # Calculate Markov Markowitz Portfolio Vals
            filename_ending = time_interval.lower()+'_'+clustering_model+'_'+str(num_clusters)+'_clusters_' + ('insamp' if in_sample else 'outsamp')
            mm_weights_df = pd.read_csv(mm_weights_dir+'markov_markowitz_weights_'+filename_ending+'.csv', index_col=0, parse_dates=True)
            mm_port_vals_df = get_portfolio_vals(mm_weights_df, test_assets_returns_df)
            mm_port_vals_df.to_csv(mm_port_vals_dir + 'markov_markowitz_port_vals_' + filename_ending +'.csv')
            
            asset_pnl_df = mm_weights_df.shift(1) * test_assets_returns_df
            asset_pnl_df.to_csv(mm_port_vals_dir + 'markov_markowitz_port_pnl_by_asset_' + filename_ending +'.csv')
            
            # Plot Markov Markowitz Portfolio
            plt.figure(figsize=(16,8))
            plt.plot(mm_port_vals_df.index, np.log(mm_port_vals_df['Port_Vals']), label='Markov Markowitz')
            plt.plot(tan_port_vals.index, np.log(tan_port_vals), label='Tangency Portfolio')
            plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX')
            plt.plot(eq_weights_port_vals.index, np.log(eq_weights_port_vals), label='Equal Weights')
            plt.title(assets_set_abr + ' Markov Markowitz Portfolio to Baselines')
            plt.xlabel('Year')
            plt.ylabel('Log Return')
            plt.legend(loc='upper left')
            plt.savefig(mm_plot_dir + 'markov_markowitz_port_vals_' + filename_ending +'.png')
            plt.close()
            
            # Save Markov Markowitz metrics
            mm_port_metrics = calc_portfolio_metrics(mm_port_vals_df['Port_Vals'])
            mm_port_metrics_df = pd.DataFrame({'Markov_Markowitz' : mm_port_metrics}).T
            mm_port_metrics_df.to_csv(mm_metrics_dir + 'markov_markowitz_port_metrics_' + filename_ending + '.csv')
            
            if not in_sample:
                subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
                port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
                port_vals_df = port_vals_df[['Markowitz_Tan_Port_'+(time_interval.lower() if time_interval[:8] != 'Rolling_' else time_interval[8:]), 'SPX', 'Eq_Weights']]
                
                rfs = read_in_rf(port_vals_df.index)
                alpha_pval_df = alpha_regression_ttest_baseline_port_vals(port_vals_df, mm_port_vals_df['Port_Vals'], rfs)
                alpha_pval_df.to_csv(mm_metrics_dir + 'markov_markowitz_alpha_ttest_' + filename_ending + '.csv')
            
    print('Simulate Markov Run Time:', round((time.time() - start_time)/60, 1), 'mins')
    
def plot_markov_markowitz(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, test_start_date, plot_spx_benchmark=False, vol_target=True):
    start_time = time.time()
    assets_set_abr = get_assets_set_abr(assets_set)
    
    for clustering_model in clustering_models:
        for num_clusters in num_clusters_list:
            # print(clustering_model, num_clusters)

            assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
            end_year = assets_returns_df.index[-1].year
            subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
            port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
            tan_port_vals = port_vals_df['Markowitz_Tan_Port_'+(time_interval.lower() if assets_set not in ['Growth_Val_Mkt_Cap', 'GVMC'] else 'yearly')]
            spx_port_vals = port_vals_df['SPX']
            eq_weights_port_vals = port_vals_df['Eq_Weights']
            
            weights_dir, plot_dir, port_vals_dir, metrics_dir = ['../Portfolios/'+assets_set+'/Versions/'+subfolder_name+ending_dir for ending_dir in ['/Weights/', '/Plots/', '/Portfolio_Vals/', '/Metrics/']]
            mm_subfolder = 'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
            mm_plot_dir = plot_dir + mm_subfolder
            mm_port_vals_dir = port_vals_dir + mm_subfolder
            mm_metrics_dir = metrics_dir + mm_subfolder
            
            filename_ending = time_interval.lower()+'_'+clustering_model+'_'+str(num_clusters)+'_clusters_outsamp'
            mm_port_vals_df = pd.read_csv(mm_port_vals_dir + 'markov_markowitz_port_vals_'+ filename_ending +'.csv', index_col=0, parse_dates=True)
            
            mm_port_vals = mm_port_vals_df['Port_Vals']
            if vol_target:
                mm_port_rets = mm_port_vals.pct_change().fillna(0)
                eq_weights_port_rets = eq_weights_port_vals.pct_change().fillna(0)
                mm_port_rets *= eq_weights_port_rets.std() / mm_port_rets.std()
                mm_port_vals = (1 + mm_port_rets).cumprod()
            
            # Plot portfolios
            plt.figure(figsize=(10,6))
            if plot_spx_benchmark:
                plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX')
            plt.plot(eq_weights_port_vals.index, np.log(eq_weights_port_vals), label='Equal Weights')
            plt.plot(tan_port_vals.index, np.log(tan_port_vals), label='Tangency Portfolio')
            plt.plot(mm_port_vals_df.index, np.log(mm_port_vals), label='Markov Markowitz', c='black')
            plt.title(assets_set_abr + ' Markov Markowitz Portfolio to Benchmarks')
            plt.xlabel('Year')
            plt.ylabel('Log Return')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles=handles[::-1], labels=labels[::-1], loc='upper left')
            plt.savefig(mm_plot_dir + 'markov_markowitz_port_vals_' + filename_ending +'.png', dpi=200)
            plt.close()
            
            # Save Portfolio Metrics
            port_metrics_df = pd.DataFrame({'Markov'   : calc_portfolio_metrics(mm_port_vals),
                                            'Tangency' : calc_portfolio_metrics(tan_port_vals),
                                            'Eq'       : calc_portfolio_metrics(eq_weights_port_vals),
                                            'S&P500'   : calc_portfolio_metrics(spx_port_vals) }).T
            port_metrics_df.to_csv(mm_metrics_dir + 'markov_to_baseline_port_metrics.csv')
    print('Plot Levered Markov Markowitz Runtime:', round((time.time() - start_time)/(60), 1), 'mins')
    
def plot_mult_markov_markowitz(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, test_start_date, plot_spx_benchmark=False, vol_target=True):
    start_time = time.time()
    assets_set_abr = get_assets_set_abr(assets_set)
    
    num_cluster_cmap = {
        3 : 'royalblue',
        4 : 'navy',
        5 : 'mediumseagreen',
        }
    
    for clustering_model in clustering_models:
        # print(clustering_model, num_clusters)

        assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
        end_year = assets_returns_df.index[-1].year
        subfolder_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
        port_vals_df = pd.read_csv(subfolder_dir+f'Portfolio_Vals/portfolio_vals_{test_start_date.year}to{end_year}.csv', index_col=0, parse_dates=True)
        tan_port_vals = port_vals_df['Markowitz_Tan_Port_'+(time_interval.lower() if assets_set not in ['Growth_Val_Mkt_Cap', 'GVMC'] else 'yearly')]
        spx_port_vals = port_vals_df['SPX']
        eq_weights_port_vals = port_vals_df['Eq_Weights']
        
        port_metrics_df = pd.DataFrame({'Tangency' : calc_portfolio_metrics(tan_port_vals),
                                        'Eq'       : calc_portfolio_metrics(eq_weights_port_vals),
                                        'S&P500'   : calc_portfolio_metrics(spx_port_vals) })
        
        # Plot portfolios
        plt.figure(figsize=(10,6))
        
        mm_cluster_subfolder = 'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'
        for num_clusters in num_clusters_list:
        
            weights_dir, plot_dir, port_vals_dir, metrics_dir = ['../Portfolios/'+assets_set+'/Versions/'+subfolder_name+ending_dir for ending_dir in ['/Weights/', '/Plots/', '/Portfolio_Vals/', '/Metrics/']]
            mm_subfolder = 'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
            mm_plot_dir = plot_dir + mm_cluster_subfolder
            mm_port_vals_dir = port_vals_dir + mm_subfolder
            mm_metrics_dir = metrics_dir + mm_cluster_subfolder
            
            filename_ending = time_interval.lower()+'_'+clustering_model+'_'+str(num_clusters)+'_clusters_outsamp'
            mm_port_vals_df = pd.read_csv(mm_port_vals_dir + 'markov_markowitz_port_vals_'+ filename_ending +'.csv', index_col=0, parse_dates=True)
            
            mm_port_vals = mm_port_vals_df['Port_Vals']
            if vol_target:
                mm_port_rets = mm_port_vals.pct_change().fillna(0)
                eq_weights_port_rets = eq_weights_port_vals.pct_change().fillna(0)
                mm_port_rets *= eq_weights_port_rets.std() / mm_port_rets.std()
                mm_port_vals = (1 + mm_port_rets).cumprod()
            
            plt.plot(mm_port_vals_df.index, np.log(mm_port_vals), label='Markov Markowitz ' + str(num_clusters) + ' Clusters (Ours)', c=num_cluster_cmap[num_clusters])
            port_metrics_df['Markov_'+str(num_clusters)] = calc_portfolio_metrics(mm_port_vals)
        
        if plot_spx_benchmark:
            plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX', c='saddlebrown')
        plt.plot(eq_weights_port_vals.index, np.log(eq_weights_port_vals), label='Equal Weights', c='firebrick')
        plt.plot(tan_port_vals.index, np.log(tan_port_vals), label='Tangency Portfolio', c='darkorange')
        plt.title(assets_set_abr + ' Markov Markowitz Portfolio to Benchmarks')
        plt.xlabel('Year')
        plt.ylabel('Log Return')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, loc='upper left')
        filename_ending = time_interval.lower()+'_'+clustering_model+'_mult_clusters_outsamp'
        plt.savefig(mm_plot_dir + 'markov_markowitz_port_vals_' + filename_ending +'.png', dpi=200)
        plt.close()
        
        # Save Portfolio Metrics
        port_metrics_df.T.to_csv(mm_metrics_dir + 'markov_to_baseline_port_metrics.csv')
    print('Plot Mult Levered Markov Markowitz Runtime:', round((time.time() - start_time)/(60), 1), 'mins')
    
def aggregate_markov_markowitz_metrics(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, in_sample=True):
    # port_metric_names = ['Sharpe', 'Sortino', 'Max_Drawdown', 'Mean_Daily_Return', 'SD_Daily_Return', 'Annual_Return', 'Annual_SD']
    port_metric_names = ['Sharpe', 'Sharpe_Sub_Avg_RF', 'Sharpe_No_RF', 'Sortino', 'Max_Drawdown', 'Mean_Daily_Return', 'SD_Daily_Return', 'Annual_Return', 'Annual_SD']
    mm_port_metrics_df = pd.DataFrame(columns=['Time_Interval', 'Clustering_Model', 'Num_Clusters'] + port_metric_names)
    metrics_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Metrics/'
    
    for clustering_model in clustering_models:
        for num_clusters in num_clusters_list:
            mm_subfolder = 'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
            mm_metrics_dir = metrics_dir + mm_subfolder
            filename_ending = time_interval.lower()+'_'+clustering_model+'_'+str(num_clusters)+'_clusters_' + ('insamp' if in_sample else 'outsamp')
            mm_port_metrics = pd.read_csv(mm_metrics_dir + 'markov_markowitz_port_metrics_' + filename_ending + '.csv', index_col=0).loc['Markov_Markowitz']
            mm_port_metrics_df.loc[len(mm_port_metrics_df.index)] = [time_interval, clustering_model, num_clusters] + mm_port_metrics.tolist()
    mm_port_metrics_df = mm_port_metrics_df.sort_values('Sharpe', ascending=False)
    mm_port_metrics_df.to_csv(metrics_dir + 'Markov_Markowitz/'+time_interval+'/' + 'agg_markov_weights_port_metrics_'+('insamp' if in_sample else 'outsamp')+'.csv', index=False)
                
                
    