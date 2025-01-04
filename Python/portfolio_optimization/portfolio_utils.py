import pandas as pd
import numpy as np
from statsmodels.regression import linear_model
from statsmodels.api import add_constant
from scipy.stats import ttest_ind, t

from data_processing.read_in_data import read_in_rf


# Initializes the lookback period if the lookback_method is yearly, quarterly, or monthly
def init_lookback_period(lookback_method, lookback_period):
    lookback_period_by_method = {'yearly' : 252, 'quarterly' : 63, 'monthly' : 21}
    if lookback_method in lookback_period_by_method:
        return lookback_period_by_method[lookback_method]
    return lookback_period

# Scale weights to be at most an input leverage while still summing to 1
def scale_weights(weights, max_leverage, weights_sum=1, force_rescale=False):
    if np.sum(np.abs(weights)) > max_leverage or force_rescale:
        abs_neg_sum = sum(np.abs(weights[weights < 0]))
        abs_pos_sum = sum(np.abs(weights[weights > 0]))
        neg_scale_factor = max_leverage if np.isclose(abs_pos_sum, 0) else (max_leverage-1)/2 + (0.5 - weights_sum/2)
        pos_scale_factor = max_leverage if np.isclose(abs_neg_sum, 0) else (max_leverage-1)/2 + (0.5 + weights_sum/2)
        weights[weights < 0] = weights[weights < 0] * neg_scale_factor / abs_neg_sum
        weights[weights > 0] = weights[weights > 0] * pos_scale_factor / abs_pos_sum
    return weights

# Calculates common portfolio metrics
def calc_portfolio_metrics(port_vals):
     port_rets = port_vals.pct_change()
     rfs = read_in_rf(port_rets.index)
     excess_rets = port_rets - rfs
     mean_excess_ret, mean_port_ret, sd_excess_ret, sd_port_ret = excess_rets.mean(), port_rets.mean(), excess_rets.std(), port_rets.std()
     sharpe = mean_excess_ret/sd_port_ret * np.sqrt(252)
     sharpe_sub_avg_rf = (mean_port_ret - rfs.mean())/sd_port_ret * np.sqrt(252)
     sharpe_no_rf_sub = mean_port_ret/sd_port_ret * np.sqrt(252)
     downside_sd_port_ret = excess_rets[excess_rets < 0].std()
     sortino = mean_excess_ret/downside_sd_port_ret * np.sqrt(252)
     roll_max = port_vals.cummax()
     daily_drawdown = port_vals/roll_max - 1
     max_daily_drawdown = min(daily_drawdown.cummin().dropna())*100
     annual_ret = (1+mean_port_ret)**252-1
     annual_sd = np.sqrt((sd_port_ret**2 + (1+mean_port_ret)**2)**252 - ((1+mean_port_ret)**2)**252)
     # annual_ret_before_fees = (1+np.mean(portfolio_vals_before_fees_over_time[portfolio_name].pct_change()))**252-1
     # annual_cost = annual_ret_before_fees - annual_ret
     return pd.Series([sharpe, sharpe_sub_avg_rf, sharpe_no_rf_sub, sortino, max_daily_drawdown, mean_port_ret, sd_port_ret, annual_ret, annual_sd], # annual_ret_before_fees, annual_cost], 
                       index=['Sharpe', 'Sharpe_Sub_Avg_RF', 'Sharpe_No_RF', 'Sortino', 'Max_Drawdown', 'Mean_Daily_Return', 'SD_Daily_Return', 'Annual_Return', 'Annual_SD'])
                                                               # 'Annual_Return_Before_Fees', 'Annual_Cost'])
                                                               
def alpha_regression_ttest(x, y, rfs, alpha_method='standard', annualize=True):
    x = x.loc[y.index].copy()
    rfs = rfs.loc[y.index].copy()
    if alpha_method == 'standard':
        model = linear_model.OLS((y-rfs).values, add_constant((x-rfs).values))
        res = model.fit()
        alpha, pval = res.params[0], 1 - t.cdf(res.tvalues[0], len(y))
    elif alpha_method == 'jensens':
        model = linear_model.OLS((y-rfs).values, (x-rfs).values)
        res = model.fit()
        beta = res.params[0]
        jensens_alpha = (y-rfs).values - rfs - beta*((x-rfs).values - rfs)
        alpha = np.mean(jensens_alpha)
        pval = ttest_ind(jensens_alpha, np.repeat(0, len(jensens_alpha))).pvalue
    else:
        raise ValueError('Invalid alpha method: {alpha_method}')
    alpha = (1+alpha)**252 - 1 if annualize else alpha
    return pd.Series({'alpha' : alpha, 'pval' : pval})

def alpha_regression_ttest_baseline_chgs(baseline_chgs_df, model_chgs, rfs, annualize=True):
    standard_alpha_pval_df = baseline_chgs_df.apply(alpha_regression_ttest, y=model_chgs, rfs=rfs, alpha_method='standard', annualize=annualize)
    jensens_alpha_pval_df = baseline_chgs_df.apply(alpha_regression_ttest, y=model_chgs, rfs=rfs, alpha_method='jensens', annualize=annualize)
    jensens_alpha_pval_df.index = ['jensens_alpha', 'jensens_pval']
    alpha_pval_df = pd.concat([standard_alpha_pval_df, jensens_alpha_pval_df])
    return alpha_pval_df

def alpha_regression_ttest_baseline_port_vals(baseline_port_vals_df, model_port_vals, rfs, annualize=True):
    baseline_chgs_df = baseline_port_vals_df.pct_change().iloc[1:]
    model_chgs = model_port_vals.pct_change().iloc[1:]
    alpha_pval_df = alpha_regression_ttest_baseline_chgs(baseline_chgs_df, model_chgs, rfs.iloc[1:], annualize=annualize)
    return alpha_pval_df

def get_assets_set_abr(assets_set):
    abr_by_assets_set = {'Growth_Val_Mkt_Cap'  : 'GVMC',
                         'SPDR_ETF_Sectors'    : 'Sectors',
                         'Developed_Markets' : 'Dev Mkts'}
    if assets_set in abr_by_assets_set:
        return abr_by_assets_set[assets_set]
    else:
        return assets_set
    
    