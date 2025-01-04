import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from pandas_datareader import data as pdr
from create_mkt_forecast_data import get_quarter
import matplotlib.pyplot as plt
import os
import time

from portfolio_optimization.portfolio_utils import get_portfolio_metrics

def get_lower_upper_chg_sharpe_by_period_df(ticker, start_date, end_date, upper_quantile, time_intervals):
    def sharpe(x):
        x_chg = x.pct_change()[1:]
        return x_chg.mean()/x_chg.std() * np.sqrt(252)
    spx_adj_close_1928to1990 = pdr.get_data_yahoo(ticker, start_date, end_date)['Adj Close'].copy()
    def get_lower_upper_chg_sharpe_for_period(adj_close, time_period, upper_quantile):
        adj_close_copy = adj_close.copy()
        if(time_period == 'Yearly'):
            adj_close_copy.index = adj_close_copy.index.year
        elif(time_period == 'Quarterly'):
            adj_close_copy.index = [get_quarter(date) for date in adj_close_copy.index]
        elif(time_period == 'Monthly'):
            adj_close_copy.index = [dt.datetime(date.year, date.month, 1) for date in adj_close_copy.index]
        else:
            raise ValueError(time_period, 'is not a valid time period')
        avg_chg_by_interval = adj_close_copy.pct_change()[1:].groupby(level=0).mean()
        sharpe_by_interval = adj_close_copy.groupby(level=0).apply(sharpe)
        lower_chg, upper_chg = avg_chg_by_interval.quantile(1-upper_quantile), avg_chg_by_interval.quantile(upper_quantile)
        lower_sharpe, upper_sharpe = sharpe_by_interval.quantile(1-upper_quantile), sharpe_by_interval.quantile(upper_quantile)
        return [lower_chg, upper_chg, lower_sharpe, upper_sharpe]
    
    lower_upper_chg_sharpe_df = pd.DataFrame(columns=['Min_Avg_Chg', 'Max_Avg_Chg', 'Min_Sharpe', 'Max_Sharpe'])
    for time_interval in time_intervals:
        lower_upper_chg_sharpe_df.loc[time_interval] = get_lower_upper_chg_sharpe_for_period(spx_adj_close_1928to1990, time_interval, upper_quantile)
    return lower_upper_chg_sharpe_df

def backtest_forecasts_spx_pos(portfolio_name, start_date, end_date, test_start_date, run_continuous=True, run_categorical=True, run_binary=True):
    
    continuous_predictands = ['SPX_Avg_Chg',  'SPX_Sharpe'] if run_continuous else []
    categorical_predictands = ['SPX_Chg_Bin', 'SPX_Sharpe_Bin'] if run_categorical else []
    binary_predictands = ['SPX_Up_Down'] if run_binary else []
    continuous_models = ['OLS', 'Stepwise', 'LASSO', 'ARIMA', 'AR']
    categorical_models = ['CART']# , 'Logistic']
    binary_models = categorical_models
    
    interpolations = [1/2, 1, 2] # 1/2 = square root 1 = linear, 2 = quadratic
    interpolation_names = {1/2 : 'square_root', 1 : 'linear', 2 : 'quadratic'}
    upper_quantiles = [0.55, 0.6, 0.75]
    small_move_invest_weights = [0.25, 0.5, 0.75]
    
    forecast_dir = '../Portfolios/'+portfolio_name+'/Mkt_Forecast/'
    
    yf.pdr_override()
    spx_df = pdr.get_data_yahoo('^SP500TR', test_start_date, end_date)[['Adj Close']].copy()
    spx_port_vals = (1 + spx_df['Adj Close'].pct_change()).cumprod()[1:]
    spx_df['Chg'] = spx_df['Adj Close'].pct_change()
    spx_df['Year'] = spx_df.index.year
    spx_df['Quarter'] = [get_quarter(date) for date in spx_df.index]
    spx_df['Month'] = [dt.datetime(date.year, date.month, 1) for date in spx_df.index]
    
    time_intervals = ['Yearly', 'Quarterly', 'Monthly']
    interpolations = []
    small_move_invest_weights = []
    time_intervals = ['Monthly']
    start_time = time.time()
    
    port_metrics_df = pd.DataFrame(columns=['Time_Interval', 'Predictand', 'Model', 'Features', 'Interpolation', 'Upper_Quantile_for_Scaling', 'Small_Move_Invest_Weight', 
                                            'Sharpe', 'Sortino', 'Max_Drawdown', 'Mean_Daily_Return', 'SD_Daily_Return', 'Annual_Return', 'Beta'])
    port_metrics_df.loc[0] = ['All', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + get_portfolio_metrics(spx_port_vals, spx_port_vals).to_list()
                
    # Continuous Backtest Forecast
    for interpolation in interpolations:
        for upper_quantile in upper_quantiles:
            upper_quantile_name = interpolation_names[interpolation]+'_interp_upper_quantile'+str(int(upper_quantile*100))
            lower_upper_chg_sharpe_df = get_lower_upper_chg_sharpe_by_period_df('^SP500TR', dt.datetime(1928, 1, 1),  dt.datetime(1989, 12, 31), upper_quantile, time_intervals)
            
            # Backtest
            for time_interval in time_intervals:
                for continuous_predictand in continuous_predictands:
                    port_vals_df = pd.DataFrame()
                    forecast_weights_df = pd.DataFrame()
                    plt.figure(figsize=(16,8))
                    plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX', c='black')
                    for continuous_model in continuous_models:
                        for predictors_name in ['EF_Coefs', 'Tech_Indicators', 'FF_Factors'] if continuous_model in ['OLS', 'Stepwise', 'LASSO'] else ['Autoregression']:
                            forecast_predictand_dir = forecast_dir+time_interval+'/Forecasts/'+continuous_predictand+'/'
                            cur_forecast_df = pd.read_csv(forecast_predictand_dir+continuous_predictand+'_'+continuous_model+'_'+predictors_name+'_forecasts.csv', index_col=0, parse_dates=(time_interval != 'Yearly'))
                            forecast_weights = pd.Series(dtype='float64')
                            
                            def get_continuous_backtest_returns(date):
                                cur_time_period_forecast = cur_forecast_df.loc[spx_df.loc[date, time_interval[:-2]] , 'Forecast']
                                if(cur_time_period_forecast >= 0):
                                    norm_denom_name = 'Max_Avg_Chg' if continuous_predictand == 'SPX_Avg_Chg' else 'Max_Sharpe'
                                    forecast_weight = min((cur_time_period_forecast / lower_upper_chg_sharpe_df.loc[time_interval, norm_denom_name])**(1/interpolation), 1)
                                    forecast_weights.loc[date] = forecast_weight
                                    return forecast_weight * (1 + spx_df.loc[date, 'Chg']) + (1-forecast_weight) - 1
                                else:
                                    norm_denom_name = 'Min_Avg_Chg' if continuous_predictand == 'SPX_Avg_Chg' else 'Min_Sharpe'
                                    forecast_weight = min((cur_time_period_forecast / lower_upper_chg_sharpe_df.loc[time_interval, norm_denom_name])**(1/interpolation), 1)
                                    forecast_weights.loc[date] = forecast_weight
                                    return forecast_weight * (1 - spx_df.loc[date, 'Chg']) + (1-forecast_weight) - 1
                            port_vals = np.maximum(pd.Series((pd.Series(spx_df.index[1:]).apply(get_continuous_backtest_returns)+1).cumprod().values, index=spx_df.index[1:]), 0)
                            port_metrics_df.loc[len(port_metrics_df.index)] = [time_interval, continuous_predictand, continuous_model, predictors_name, interpolation_names[interpolation], 
                                                                                 upper_quantile, np.nan] + get_portfolio_metrics(port_vals, spx_port_vals).to_list()
                            
                            strategy_name = time_interval+'_'+continuous_predictand+'_'+continuous_model+'_'+predictors_name
                            port_vals_df[strategy_name] = port_vals
                            plt.plot(port_vals_df.index, np.log(port_vals_df[strategy_name]), label=continuous_model + ' ' + predictors_name)
                            forecast_weights_df[continuous_model+'_'+predictors_name] = forecast_weights
                    # Save forecast weights
                    forecast_weights_dir = forecast_dir + 'Forecast_Backtest/Forecast_Weights/' + upper_quantile_name + '/'
                    if not os.path.exists(forecast_weights_dir):
                        os.makedirs(forecast_weights_dir)
                    forecast_weights_df.to_csv(forecast_weights_dir + time_interval + '_' + continuous_predictand + '_forecast_weights.csv')
                    
                    # Save performance vals 
                    backtest_performance_dir = forecast_dir + 'Forecast_Backtest/Performance_Vals/' + upper_quantile_name + '/'
                    if not os.path.exists(backtest_performance_dir):
                        os.makedirs(backtest_performance_dir)
                    port_vals_df.to_csv(backtest_performance_dir + time_interval + '_' + continuous_predictand + '_forecast_models_performance_over_time.csv')
                    
                    # Save backtest plots
                    plt.title(time_interval + ' ' + continuous_predictand + ' Forecast Models Performance' )
                    plt.ylabel('Log Return')
                    plt.xlabel('Time')
                    plt.legend(loc='upper left')
                    backtest_plot_dir = forecast_dir + 'Forecast_Backtest/Plots/' + upper_quantile_name + '/'
                    if not os.path.exists(backtest_plot_dir):
                        os.makedirs(backtest_plot_dir)
                    plt.savefig(backtest_plot_dir + time_interval+'_'+continuous_predictand+'_forecast_backtest_performance_over_time_plot.png')
                    plt.close()
    
    # Binary Forecast Backtest
    for time_interval in time_intervals:
        for binary_predictand in binary_predictands:
            port_vals_df = pd.DataFrame()
            plt.figure(figsize=(16,8))
            plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX', c='black')
            for binary_model in binary_models:
                for predictors_name in ['EF_Coefs', 'Tech_Indicators', 'FF_Factors']:
                    forecast_predictand_dir = forecast_dir+time_interval+'/Forecasts/'+binary_predictand+'/'
                    cur_forecast_df = pd.read_csv(forecast_predictand_dir+binary_predictand+'_'+binary_model+'_'+predictors_name+'_forecasts.csv', index_col=0, parse_dates=(time_interval != 'Yearly'))
                        
                    def get_binary_backtest_returns(date):
                        cur_time_period_forecast = cur_forecast_df.loc[spx_df.loc[date, time_interval[:-2]] , 'Forecast']
                        if(cur_time_period_forecast == 'Up'):
                            return spx_df.loc[date, 'Chg']
                        else:
                            return - spx_df.loc[date, 'Chg']
                    port_vals = np.maximum(pd.Series((pd.Series(spx_df.index[1:]).apply(get_binary_backtest_returns)+1).cumprod().values, index=spx_df.index[1:]), 0)
                    port_metrics_df.loc[len(port_metrics_df.index)] = [time_interval, binary_predictand, binary_model, predictors_name, np.nan, 
                                                                       np.nan, np.nan] + get_portfolio_metrics(port_vals, spx_port_vals).to_list()
                    strategy_name = time_interval+'_'+binary_predictand+'_'+binary_model+'_'+predictors_name
                    port_vals_df[strategy_name] = port_vals
                    plt.plot(port_vals_df.index, np.log(port_vals_df[strategy_name]), label=binary_model + ' ' + predictors_name)
            # Save performance vals 
            backtest_performance_dir = forecast_dir + 'Forecast_Backtest/Performance_Vals/Binary/'
            if not os.path.exists(backtest_performance_dir):
                os.makedirs(backtest_performance_dir)
            port_vals_df.to_csv(backtest_performance_dir + time_interval + '_' + binary_predictand + '_forecast_models_performance_over_time.csv')
            
            # Save backtest plots
            plt.title(time_interval + ' ' + binary_predictand + ' Forecast Models Performance' )
            plt.ylabel('Log Return')
            plt.xlabel('Time')
            plt.legend(loc='upper left')
            backtest_plot_dir = forecast_dir + 'Forecast_Backtest/Plots/Binary/'
            if not os.path.exists(backtest_plot_dir):
                os.makedirs(backtest_plot_dir)
            plt.savefig(backtest_plot_dir + time_interval+'_'+binary_predictand+'_forecast_backtest_performance_over_time_plot.png')
            plt.close()
    
    # Categorical Forecast Backtest
    for small_move_invest_weight in small_move_invest_weights:
        small_move_weight_name = 'small_move_invest_weight'+str(int(small_move_invest_weight*100))
        for time_interval in time_intervals:
            for categorical_predictand in categorical_predictands:
                port_vals_df = pd.DataFrame()
                plt.figure(figsize=(16,8))
                plt.plot(spx_port_vals.index, np.log(spx_port_vals), label='SPX', c='black')
                for categorical_model in categorical_models:
                    for predictors_name in ['EF_Coefs', 'Tech_Indicators', 'FF_Factors']:
                        forecast_predictand_dir = forecast_dir+time_interval+'/Forecasts/'+categorical_predictand+'/'
                        cur_forecast_df = pd.read_csv(forecast_predictand_dir+categorical_predictand+'_'+categorical_model+'_'+predictors_name+'_forecasts.csv', index_col=0, parse_dates=(time_interval != 'Yearly'))
                            
                        def get_categorical_backtest_returns(date):
                            cur_time_period_forecast = cur_forecast_df.loc[spx_df.loc[date, time_interval[:-2]] , 'Forecast']
                            if(cur_time_period_forecast == 'High'):
                                return spx_df.loc[date, 'Chg']
                            elif(cur_time_period_forecast == 'Med_High'):
                                return small_move_invest_weight * (1 + spx_df.loc[date, 'Chg']) + (1-small_move_invest_weight) - 1
                            elif(cur_time_period_forecast == 'Med_Low'):
                                return small_move_invest_weight * (1 - spx_df.loc[date, 'Chg']) + (1-small_move_invest_weight) - 1
                            elif(cur_time_period_forecast == 'Low'):
                                return - spx_df.loc[date, 'Chg']
                        port_vals = np.maximum(pd.Series((pd.Series(spx_df.index[1:]).apply(get_categorical_backtest_returns)+1).cumprod().values, index=spx_df.index[1:]), 0)
                        port_metrics_df.loc[len(port_metrics_df.index)] = [time_interval, categorical_predictand, categorical_model, predictors_name, np.nan, 
                                                                       np.nan, small_move_invest_weight] + get_portfolio_metrics(port_vals, spx_port_vals).to_list()
                        strategy_name = time_interval+'_'+categorical_predictand+'_'+categorical_model+'_'+predictors_name
                        port_vals_df[strategy_name] = port_vals
                        plt.plot(port_vals_df.index, np.log(port_vals_df[strategy_name]), label=categorical_model + ' ' + predictors_name)
                # Save performance vals 
                backtest_performance_dir = forecast_dir + 'Forecast_Backtest/Performance_Vals/' + small_move_weight_name + '/'
                if not os.path.exists(backtest_performance_dir):
                    os.makedirs(backtest_performance_dir)
                port_vals_df.to_csv(backtest_performance_dir + time_interval + '_' + categorical_predictand + '_forecast_models_performance_over_time.csv')
                
                # Save backtest plots
                plt.title(time_interval + ' ' + categorical_predictand + ' Forecast Models Performance' )
                plt.ylabel('Log Return')
                plt.xlabel('Time')
                plt.legend(loc='upper left')
                backtest_plot_dir = forecast_dir + 'Forecast_Backtest/Plots/'  + small_move_weight_name + '/'
                if not os.path.exists(backtest_plot_dir):
                    os.makedirs(backtest_plot_dir)
                plt.savefig(backtest_plot_dir + time_interval+'_'+categorical_predictand+'_forecast_backtest_performance_over_time_plot.png')
                plt.close()
                    
    # Save metrics
    backtest_metrics_dir = forecast_dir + 'Metrics/'
    if not os.path.exists(backtest_metrics_dir):
        os.makedirs(backtest_metrics_dir)
    port_metrics_df.to_csv(backtest_metrics_dir + 'mkt_forecast_backtest_metrics.csv', index=False)
    print('Total Forecast Models runtime:', str(round((time.time() - start_time)/60, 1)), 'mins')
        
        
                
                
                