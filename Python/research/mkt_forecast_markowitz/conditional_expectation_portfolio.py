import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt
import time
import os
from sklearn.linear_model import LinearRegression

from data_processing.read_in_data import read_in_rf, read_in_eq_weight
from data_processing.ts_utils import get_month_first_date, get_prev_months_first_date
from portfolio_optimization.markowitz import calc_all_markowitz_weights_monthly, calc_all_markowitz_weights_daily_rolling

'''
Calculates the markowitz tangency portfolio weights with
expected returns conditional on a probabistic forecast of 
whether the market will be up or down
'''

def calc_beta_alpha(asset_rets, mkt_rets, rf, use_alpha=True):
    regr = LinearRegression(fit_intercept=use_alpha).fit((mkt_rets-rf).values.reshape(-1,1), (asset_rets-rf).values)
    beta, alpha, r2 = regr.coef_[0], (regr.intercept_ if use_alpha else 0), r2_score(regr, (mkt_rets-rf).values.reshape(-1,1), (asset_rets-rf).values)
    return beta, alpha, r2

def r2_score(regr, X, y):
    return pd.DataFrame({'y' : y, 'y_pred' : regr.predict(X)}).corr().iloc[0,1]**2

def filter_above_below(df, thresholds, above=True):
    filtered_df = df.copy()
    for col in filtered_df.columns:
        remove_mask = filtered_df[col] < thresholds[col] if above else filtered_df[col] > thresholds[col]
        filtered_df.loc[remove_mask, col] = np.nan
    return filtered_df

def cond_exp_return(assets_returns_df, full_upto_assets_returns_df, mkt_forecast_df, rfs, mkt_returns, window=None, beta_lookback=None, 
                    discretize_prob=True, weight_capm=True, up_forecast=True):
    is_rolling = window is not None
    if beta_lookback is None:
        beta_lookback = 252 if is_rolling else 12
    cur_date = assets_returns_df.index[-1] if is_rolling else dt.datetime(assets_returns_df.index[0].year, assets_returns_df.index[0].month, 1)
    mkt_returns, rfs, full_upto_assets_returns_df = mkt_returns.copy(), rfs.copy(), full_upto_assets_returns_df.copy()
    if not is_rolling:
        mkt_returns.index = [dt.datetime(date.year, date.month, 1) for date in mkt_returns.index]
        rfs.index = [dt.datetime(date.year, date.month, 1) for date in rfs.index]
        full_upto_assets_returns_df.index = [dt.datetime(date.year, date.month, 1) for date in full_upto_assets_returns_df.index]
    # cur_forecast = mkt_forecast_df.shift(-1).loc[cur_date, 'Forecast']
    cur_pred_prob = mkt_forecast_df.loc[cur_date, 'Prob']
    if not up_forecast:
        cur_pred_prob = 1 - cur_pred_prob

    cur_mkt_returns = mkt_returns.iloc[-window:] if is_rolling else mkt_returns.loc[cur_date]
    if not is_rolling:
        beta_lookback_dates = [cur_date] + get_prev_months_first_date(cur_date, beta_lookback-1) 
        assets_returns_df_beta_lookback = full_upto_assets_returns_df.loc[beta_lookback_dates].copy()
        mkt_returns_beta_lookback = mkt_returns.loc[beta_lookback_dates]
    else:
        assets_returns_df_beta_lookback = full_upto_assets_returns_df.iloc[-window:].copy()
        mkt_returns_beta_lookback = mkt_returns.iloc[-window:]
    cur_rf = rfs.loc[[cur_date]].iloc[-1]
    beta_alpha_df = assets_returns_df_beta_lookback.apply(calc_beta_alpha, axis=0, args=(mkt_returns_beta_lookback, cur_rf), use_alpha=False)
    beta_alpha_df.index = ['beta', 'alpha', 'R2']
    
    mu_mkt = cur_mkt_returns.mean()
    sd_mkt = cur_mkt_returns.std()
    discrete_cur_pred_prob = 1 if cur_pred_prob > 0.5 else 0
    c = discrete_cur_pred_prob if discretize_prob else cur_pred_prob
    conditional_expected_mkt_ret = mu_mkt + sd_mkt * (2*c-1)*sp.stats.norm.pdf(-mu_mkt/sd_mkt) / (c-(2*c-1)*sp.stats.norm.cdf(-mu_mkt/sd_mkt))
    conditional_expected_assets_rets = beta_alpha_df.loc['beta',:] * (conditional_expected_mkt_ret - cur_rf) + cur_rf # + beta_alpha_df.loc['alpha',:]
    if weight_capm:
        conditional_expected_assets_rets = beta_alpha_df.loc['R2',:] * conditional_expected_assets_rets + (1 - beta_alpha_df.loc['R2',:]) * assets_returns_df.mean()
    return conditional_expected_assets_rets

def find_all_cond_exp_mark_tan_port_weights(assets_returns_df, mkt_forecast_df, rfs, mkt_returns, test_start_date, 
                                            weights_sum_list=[-1, 1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, window=None,
                                            discretize_prob=True, weight_capm=True, up_forecast=True):
    mean_est_kwargs = {'mkt_forecast_df' : mkt_forecast_df, 
                       'rfs'             : rfs, 
                       'mkt_returns'     : mkt_returns,
                       'window'          : window,
                       'discretize_prob' : discretize_prob,
                       'weight_capm'     : weight_capm,
                       'up_forecast'     : up_forecast}
    if window is not None:
        test_assets_returns_df = assets_returns_df[assets_returns_df.index >= test_start_date]
        cond_exp_tan_port_weights_df = calc_all_markowitz_weights_daily_rolling(test_assets_returns_df.iloc[:-1], window, rfs, portfolio_selection='tangency', 
                                                                                mean_estimator=cond_exp_return, mean_est_kwargs=mean_est_kwargs,
                                                                                weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound,
                                                                                max_leverage=max_leverage, max_leverage_method=max_leverage_method)
    else:
        cond_exp_tan_port_weights_df = calc_all_markowitz_weights_monthly(assets_returns_df, rfs, portfolio_selection='tangency', start_date=get_month_first_date(test_start_date),
                                                                          mean_estimator=cond_exp_return, mean_est_kwargs=mean_est_kwargs,
                                                                          weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, 
                                                                          max_leverage=max_leverage, max_leverage_method=max_leverage_method)
    return cond_exp_tan_port_weights_df

def find_all_mkt_forecast_mark_tan_port_weights(assets_set, subfolder_name, time_interval, forecast_model_name, predictand, predictors_name, test_start_date, 
                                                        lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, window=None, 
                                                        discretize_prob=True, weight_capm=True, up_forecast=True):
    start_time = time.time()
    # Set up directories
    port_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/'
    mf_weights_dir = port_dir+'Weights/Mkt_Forecast/'+time_interval+'/'
    if not os.path.exists(mf_weights_dir):
        os.makedirs(mf_weights_dir)
    
    # Read in data
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    mkt_forecast_df = pd.read_csv('../Portfolios/'+assets_set+'/Mkt_Forecast/'+time_interval+'/'+predictors_name+'/'+forecast_model_name+'/Forecasts/'+predictand+'/'+predictand+'_'+forecast_model_name+'_'+predictors_name+'_forecasts.csv', index_col=0, parse_dates=True)
    rfs = read_in_rf(assets_returns_df.index)
    mkt_returns = read_in_eq_weight(assets_set, assets_returns_df.index)['Chg']

    # Calculate and save results
    cond_exp_tan_port_weights_df = find_all_cond_exp_mark_tan_port_weights(assets_returns_df, mkt_forecast_df, rfs, mkt_returns, test_start_date, 
                                                                           lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method,
                                                                           window=window, discretize_prob=discretize_prob, weight_capm=weight_capm, up_forecast=up_forecast)
    cond_exp_tan_port_weights_df.to_csv(mf_weights_dir+'daily_weights_mkt_forecast_'+forecast_model_name+'_'+predictors_name+'_'+time_interval.lower()+'.csv')
    print('Conditional Expectation Tangency Portfolio Runtime:', str(round((time.time() - start_time)/(60))), 'mins')
    return cond_exp_tan_port_weights_df
    
    
    
    