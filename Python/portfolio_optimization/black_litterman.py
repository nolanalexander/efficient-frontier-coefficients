import pandas as pd
import numpy as np

from data_processing.read_in_data import read_in_cur_price_and_outstanding_shares
from portfolio_optimization.markowitz import calc_mean_var_weights, calc_tan_port_weights

'''
Implementation of the Black-Litterman Model.  The model uses a weighted combination of the market 
capitalization views and investor's views to output better estimates of the excess expected returns
'''

inv = np.linalg.pinv

# Finds the market equilibrium portfolio weights based on market capitalization
def find_market_equilibrium_weights(assets_set, tickers, date):
    cur_value_adj_close, shares_outstanding = read_in_cur_price_and_outstanding_shares(assets_set, tickers, date)
    market_capitalization = cur_value_adj_close * shares_outstanding
    market_cap_weights = market_capitalization/market_capitalization.sum()
    return market_cap_weights

# Calculates the risk aversion factor based on the market and risk free rate data
def calculate_risk_aversion_factor(assets_returns_df, market_cap_weights, rf):
    mkt_excess_returns_df = (assets_returns_df.mul(market_cap_weights, axis=1)).sub(rf, axis=0)
    avg_mkt_excess = mkt_excess_returns_df.mean()
    sd_mkt_excess = mkt_excess_returns_df.std()
    delta = avg_mkt_excess/sd_mkt_excess**2
    return delta

# performs reverse optimization to find the equilibrium expected excess returns
def reverse_optimization(risk_aversion_factor, cov_matrix, market_equilibrium_portfolio_weights):
    delta = risk_aversion_factor
    sigma = cov_matrix
    w_mkt = market_equilibrium_portfolio_weights
    
    pi = (delta * sigma).dot(w_mkt)
    return pi

def reweight_view(view, market_equilibrium_weights):
    view_sum = view.sum()
    rescaled_view = view * market_equilibrium_weights
    if view_sum == 0:
        return rescaled_view/rescaled_view[rescaled_view > 0].sum()
    elif view_sum == 1:
        return rescaled_view/rescaled_view.sum()
    else:
        raise ValueError(f'Invalid view sum: {view_sum}')

def calc_view_unconfidence(view, cov_matrix):
    return view.dot(cov_matrix).dot(view)

'''
Calculates the Black-Litterman estimates of expected returns based on prior views and view of the market
Output (posterior): an estimate of expected returns based on views and market equilibrium,
to be input to a Markowitz Portfolio Mean-Variance Optimizer
Inputs (priors):
cov_matrix: the covariance matrix of asset growth data
view_portfolios: k x n matrix, where k = num views and n = num assets,
  each row is a view on an asset that adds up to 0 or 1
  views on the diagonal are Absolute (asset returns %)
  views not on the diagonal are Relative (row asset will outperfrom col asset by %), usually = 1
expected_returns_each_view: a column vector of expected returns for each view-portfolio, 
  e.g. np.array([0.1, 0.5, 0.3]).T refers to an 10% return in view 1, 50% return in view 2, etc.
levels_of_unconfidence: a column vector of standard deviations for expected returns
  assuming normal distribution -> 0.68 probability that realization is in interval
weight_on_views: a constant for the weight on the view and equilibrium portfolios,
  conflicting literature: some suggest close to 1, others close to 0
market_equilibrium_portfolio_weights: weights of equilibrium portfolio based on market views
'''
def calc_black_litterman_expected_returns(assets_returns_df, rf, portfolio_views, expected_returns_each_view, 
                                          weight_on_views, market_equilibrium_weights, levels_of_unconfidence=None,
                                          reweight_P=True):
    cov_matrix = assets_returns_df.cov()
    sigma = cov_matrix
    P = portfolio_views
    if reweight_P:
        P = P.apply(reweight_view, axis=1, market_equilibrium_weights=market_equilibrium_weights)
    if levels_of_unconfidence is None:
        levels_of_unconfidence = P.apply(calc_view_unconfidence, axis=1, cov_matrix=sigma)
    q_hat = np.array(expected_returns_each_view)
    omega = np.diag(levels_of_unconfidence**2)
    tau = weight_on_views
    risk_aversion_factor = calculate_risk_aversion_factor(assets_returns_df, market_equilibrium_weights, rf)
    pi = reverse_optimization(risk_aversion_factor, cov_matrix, market_equilibrium_weights)
        
    mu_star = inv(inv(tau * sigma) + P.T.dot(inv(omega)).dot(P)).dot(
              (inv(tau * sigma).dot(pi) + P.T.dot(inv(omega)).dot(q_hat)))
    return pd.Series(mu_star, index=assets_returns_df.columns)

def calc_black_litterman_cov_matrix(assets_returns_df, portfolio_views, 
                                    weight_on_views, market_equilibrium_weights=None, levels_of_unconfidence=None,
                                    reweight_P=True):
    cov_matrix = assets_returns_df.cov()
    sigma = cov_matrix
    P = portfolio_views
    if reweight_P:
        if market_equilibrium_weights is None:
            raise ValueError('market_equilibrium_weights not specified')
        P = P.apply(reweight_view, axis=1, market_equilibrium_weights=market_equilibrium_weights)
    if levels_of_unconfidence is None:
        levels_of_unconfidence = P.apply(calc_view_unconfidence, axis=1, cov_matrix=sigma)
    omega = np.diag(levels_of_unconfidence**2)
    tau = weight_on_views
    
    sigma_star = sigma + inv(inv(tau * sigma) + P.T.dot(inv(omega).dot(P)))
    return pd.DataFrame(sigma_star, columns=assets_returns_df.columns, index=assets_returns_df.columns)

# Heuristic calculation of Black-Litterman optimal weights 
def calc_black_litterman_implied_weights(black_litterman_expected_returns, black_litterman_cov_matrix, risk_aversion_factor):
    
    mu_star = black_litterman_expected_returns
    delta = risk_aversion_factor
    sigma = black_litterman_cov_matrix
    
    w_star = np.linalg.inv(delta * sigma).dot(mu_star)
    w_star = w_star/sum(np.abs(w_star))
    return w_star

def calc_black_litterman_mean_var_weights(portfolio_return, assets_returns_df, rf, portfolio_views, expected_returns_each_view, 
                                          weight_on_views, market_equilibrium_weights, levels_of_unconfidence=None,
                                          reweight_P=False, lower_bound=None, upper_bound=None, 
                                          max_leverage=1.5, max_leverage_method=None):
    mu_star = calc_black_litterman_expected_returns(assets_returns_df, rf, portfolio_views, expected_returns_each_view, 
                                                    weight_on_views, market_equilibrium_weights, levels_of_unconfidence=levels_of_unconfidence,
                                                    reweight_P=reweight_P)
    sigma_star = calc_black_litterman_cov_matrix(assets_returns_df, portfolio_views, 
                                                 weight_on_views, market_equilibrium_weights=market_equilibrium_weights, levels_of_unconfidence=levels_of_unconfidence,
                                                 reweight_P=reweight_P)
    weights = calc_mean_var_weights(portfolio_return, mu_star, sigma_star, lower_bound=lower_bound, upper_bound=upper_bound, 
                                    max_leverage=max_leverage, max_leverage_method=max_leverage_method)
    return weights
    
def calc_black_litterman_tan_port_weights(assets_returns_df, rf, portfolio_views, expected_returns_each_view, 
                                          weight_on_views, market_equilibrium_weights, levels_of_unconfidence=None,
                                          reweight_P=False, lower_bound=None, upper_bound=None, 
                                          max_leverage=1.5, max_leverage_method=None):
    mu_star = calc_black_litterman_expected_returns(assets_returns_df, rf, portfolio_views, expected_returns_each_view, 
                                                    weight_on_views, market_equilibrium_weights, levels_of_unconfidence=levels_of_unconfidence,
                                                    reweight_P=reweight_P)
    sigma_star = calc_black_litterman_cov_matrix(assets_returns_df, portfolio_views, 
                                                 weight_on_views, market_equilibrium_weights=market_equilibrium_weights, levels_of_unconfidence=levels_of_unconfidence,
                                                 reweight_P=reweight_P)
    weights = calc_tan_port_weights(mu_star, sigma_star, rf, lower_bound=lower_bound, upper_bound=upper_bound, 
                                    max_leverage=max_leverage, max_leverage_method=max_leverage_method)
    return weights






