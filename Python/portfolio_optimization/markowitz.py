import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime as dt
import os
from numpy_ext import rolling_apply, expanding_apply
# pip install numpy-ext
import cvxpy as cvx
import warnings
from itertools import product
import time

from data_processing.ts_utils import get_quarters, get_last_quarter
from portfolio_optimization.portfolio_utils import init_lookback_period, scale_weights
from portfolio_optimization.mean_estimation import mean
from portfolio_optimization.cov_estimation import cov

'''
The Markowitz Mean-Variance model includes the calcuation of 
specific portfolios on the efficient frontiers, MVP, tangency portfolio, 
and efficient frontier coefficients
'''

def calc_mean_var_weights(portfolio_return : float, assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame, 
                          optimizer='CVXPY', weights_sum=1, lower_bound=None, upper_bound=None,
                          max_leverage=1.5, max_leverage_method=None, annual=False, date=None) -> np.array:
    '''
    Calculates the Mean-Variance weights, trying multiple optimizers and methods when one fails
    portfolio_return    : the expected portfolio return to achieve, set with constraint
    assets_exp_rets     : expected return vector
    cov_matrix          : covariance matrix
    optimizer           : the optimizer to use (CVXPY or SciPy)
    weights_sum         : what the weights must sum up, depends on desired net position
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    max_leverage_method : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    annual              : whether to annualize return and variance
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    init_optimizer, init_max_leverage_method = optimizer, max_leverage_method
    if max_leverage is None:
        max_leverage_methods = [None]
    elif init_max_leverage_method != 'scaling':
        max_leverage_methods = [init_max_leverage_method, 'scaling'] 
    else:
        max_leverage_methods = ['scaling']
    optimizers = [init_optimizer, 'SciPy'] if init_optimizer != 'SciPy' else ['SciPy']
    
    for max_leverage_method, optimizer in product(max_leverage_methods, optimizers):
    
        if optimizer == 'CVXPY':
            max_leverage_methods = [init_max_leverage_method, 'scaling'] if init_max_leverage_method != 'scaling' else ['scaling']
            for max_leverage_method in max_leverage_methods:
                w = cvx.Variable(len(assets_exp_rets))
                variance = cvx.quad_form(w, np.asmatrix(cov_matrix))
                er = np.array(assets_exp_rets) @ w
                if annual:
                    variance = (variance + (1+er)**2)**252 -((1+er)**2)**252
                    er = (1+er)**252-1
                
                cons = [cvx.sum(w) == weights_sum,
                        er == portfolio_return]
                if max_leverage_method is not None and max_leverage_method == 'constraint':
                    cons += [cvx.norm(w, 1) <= max_leverage]
                cons += [w >= lower_bound] if lower_bound is not None else [w >= -100]
                cons += [w <= upper_bound] if upper_bound is not None else [w <= 100]
                prob = cvx.Problem(cvx.Minimize(variance), cons)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prob.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-15, eps_dual_inf=1e-15, max_iter=10000)
                        # Attempts 5 warm runs if not optimal
                        solve_attempts = 1
                        while prob.status != 'optimal' and solve_attempts < 5:
                            prob.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-15, eps_dual_inf=1e-15, max_iter=10000)
                            solve_attempts += 1
                        weights = w.value if prob.status == 'optimal' else None
                        status = prob.status
                except:
                    # if max_leverage_method == 'scaling':
                    #     print(f'RuntimeWarning: In M-V, CVXPY reached max_iter on {date.date() if date is not None else date} with max_leverage_method \'{max_leverage_method}\'')
                    weights = None
                    status = 'failure'
                
                if status == 'optimal':
                    break
                else:
                    weights = None
                    # print(date)
                    # prob = cvx.Problem(cvx.Minimize(variance), cons)
                    # prob.solve(verbose=True)
        
        elif optimizer == 'SciPy':
            num_assets = len(assets_exp_rets)
            weights = np.repeat(1/num_assets, num_assets)
            def variance_daily(input_weights):
                return input_weights.T @ np.array(cov_matrix) @ input_weights
            def expected_return_daily(input_weights):
                return np.array(assets_exp_rets).T @ input_weights
            def variance_yearly(input_weights):
                return ((variance_daily(input_weights) + (1+expected_return_daily(input_weights))**2)**252 
                        -((1+expected_return_daily(input_weights))**2)**252)
            def expected_return_yearly(input_weights):
                return (1+expected_return_daily(input_weights))**252-1
            
            variance, expected_return = (variance_yearly, expected_return_yearly) if annual else (variance_daily, expected_return_daily)
            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            cons.append({'type':'eq', 'fun': lambda x: portfolio_return - expected_return(x)})
            if max_leverage_method is not None and max_leverage_method == 'constraint':
                cons.append({'type':'ineq', 'fun': lambda x: max_leverage - sum(abs(x)) })
            bounds = ((lower_bound, upper_bound),)*num_assets
            
            optimization = minimize(variance, weights, method='SLSQP', bounds=bounds, 
                                    options = {'disp':False, 'ftol': 1e-15, 'maxiter': 1e4} , constraints=cons)
            if optimization.success:
                weights = optimization.x
            elif not optimization.success and max_leverage_method == 'scaling':
                weights = None
                # print(date)
                # print(optimization)
                # print(f'RuntimeWarning: In M-V, SciPy failed to converge on {date.date() if date is not None else date}. Setting weights to None.')
    # if optimizer != init_optimizer:
    #     print(f'RuntimeWarning: In M-V, {init_optimizer} solver failed on {date.date() if date is not None else date}. Defaulting to {optimizer} solver.')
    # if max_leverage_method != init_max_leverage_method and weights is not None:
    #     print(f'RuntimeWarning: In M-V, max_leverage_method \'{init_max_leverage_method}\' failed on {date.date() if date is not None else date}. Defaulting to \'{max_leverage_method}\'.')
    weights = correct_extreme_weights(weights, assets_exp_rets=assets_exp_rets, cov_matrix=cov_matrix, 
                                      max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=date)
    return weights

# Scales leverage or use MVP
def correct_extreme_weights(weights : np.array, max_leverage_method='scaling', weights_sum=1, assets_exp_rets=None, cov_matrix=None, mvp_weights=None,
                            lower_bound=None, upper_bound=None, max_leverage=1.5, date=None):
    '''
    Scales leverage or use MVP
    weights             : array-like weights to scale
    max_leverage_method : how to limit max_leverage (None, 'constraint', or 'scaling')
    weights_sum         : what the weights must sum up, depends on desired net position
    assets_exp_rets     : expected return vector used in MVP calculation
    cov_matrix          : covariance matrix used in MVP calculation
    mvp_weights         : the pre-calculated MVP weights used for 
    weights_sum         : what the weights must sum up, depends on desired net position
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    if weights is None or np.isinf(weights).any():
        max_leverage_method = 'MVP'
        print(f'RuntimeWarning: Weights is {weights} on {date}. Defaulting to MVP weights.')
        
    if max_leverage_method is None:
        pass
    elif max_leverage_method == 'constraint':
        pass
    elif max_leverage_method == 'MVP':
        if weights is None or np.isinf(weights).any() or np.sum(np.abs(weights)) > max_leverage:
            if mvp_weights is not None:
                weights = mvp_weights
            else:
                weights = calc_MVP_weights(assets_exp_rets, cov_matrix, 
                                           lower_bound=lower_bound, upper_bound=upper_bound, 
                                           max_leverage=max_leverage, max_leverage_method='constraint')
    elif max_leverage_method == 'scaling':
        weights = scale_weights(weights, max_leverage, weights_sum)
    else:
        raise ValueError(f'Undefined max leverage method: {max_leverage_method}')
    return weights

def get_ef_coefs(assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame) -> list:
    '''
    Finds the efficient frontier A, B, C coefficients
    '''
    e = np.ones(len(assets_exp_rets))
    inv_cov_mat = np.linalg.pinv(np.array(cov_matrix))
    A = e @ inv_cov_mat @ e.T
    B = np.array(assets_exp_rets) @ inv_cov_mat @ e.T
    C = np.array(assets_exp_rets) @ inv_cov_mat @ np.array(assets_exp_rets).T
    return [A, B, C]

def calc_MVP_weights(assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame, 
                     optimizer='CVXPY', weights_sum=1, lower_bound=None, upper_bound=None, 
                     max_leverage=1.5, max_leverage_method=None, date=None) -> np.array:
    '''
    Calculates the Minimum Variance Portfolio weights
    assets_exp_rets     : expected return vector
    cov_matrix          : covariance matrix
    optimizer           : the optimizer to use (CVXPY or SciPy)
    weights_sum         : what the weights must sum up, depends on desired net position
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    max_leverage_method : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    init_optimizer, init_max_leverage_method = optimizer, max_leverage_method
    max_leverage_methods = [init_max_leverage_method, 'scaling'] if init_max_leverage_method != 'scaling' else ['scaling']
    optimizers = [init_optimizer, 'SciPy'] if init_optimizer != 'SciPy' else ['SciPy']
    
    for max_leverage_method, optimizer in product(max_leverage_methods, optimizers):
            
        if optimizer == 'CVXPY':
            w = cvx.Variable(len(assets_exp_rets))
            variance = cvx.quad_form(w, np.asmatrix(cov_matrix))
            
            cons = [cvx.sum(w) == weights_sum]
            if max_leverage_method is not None and max_leverage_method == 'constraint':
                cons += [cvx.norm(w, 1) <= max_leverage]
            cons += [w >= lower_bound] if lower_bound is not None else [w >= -100]
            cons += [w <= upper_bound] if upper_bound is not None else [w <= 100]
            prob = cvx.Problem(cvx.Minimize(variance), cons)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-15, eps_dual_inf=1e-15, max_iter=10000)
                    # Attempts 5 warm runs if not optimal
                    solve_attempts = 1
                    while prob.status != 'optimal' and solve_attempts < 5:
                        prob.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-15, eps_dual_inf=1e-15, max_iter=10000)
                        solve_attempts += 1
                    weights = w.value if prob.status == 'optimal' else None
                    status = prob.status
            except:
                # if max_leverage_method == 'scaling':
                #     print(f'RuntimeWarning: In MVP, CVXPY reached max_iter on {date.date() if date is not None else date} with max_leverage_method \'{max_leverage_method}\'')
                weights = None
                status = 'failure'
            
            if status == 'optimal':
                break
            else:
                weights = None
                # print(date)
                # prob = cvx.Problem(cvx.Minimize(variance), cons)
                # prob.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-15, eps_dual_inf=1e-15, max_iter=10000)
                
        elif optimizer == 'SciPy':
            num_assets = len(assets_exp_rets)
            weights = np.repeat(1/num_assets, num_assets)
            def standev(input_weights):
                return np.sqrt(input_weights.T @ np.array(cov_matrix) @ input_weights)
            
            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            if max_leverage_method is not None and max_leverage_method == 'constraint':
                cons.append({'type':'ineq', 'fun': lambda x: max_leverage - sum(abs(x)) })
            bounds = ((lower_bound, upper_bound),)*num_assets
            optimization = minimize(standev, weights, method='SLSQP' ,
                           bounds = bounds,
                           options = {'disp':False, 'ftol': 1e-15, 'maxiter': 1e4} ,
                           constraints=cons)
            if optimization.success:
                weights = optimization.x
                break
            elif not optimization.success and max_leverage_method == 'scaling':
                weights = None
                # print(date)
                # print(optimization)
                # print(f'RuntimeWarning: In MVP, SciPy failed to converge on {date.date() if date is not None else date}. Setting weights to None.')
    # if optimizer != init_optimizer:
    #     print(f'RuntimeWarning: In MVP, {init_optimizer} solver failed on {date.date() if date is not None else date}. Defaulting to {optimizer} solver.')
    # if max_leverage_method != init_max_leverage_method and weights is not None:
    #     print(f'RuntimeWarning: In MVP, max_leverage_method \'{init_max_leverage_method}\' failed on {date.date() if date is not None else date}. Defaulting to \'{max_leverage_method}\'.')
    if max_leverage_method == 'scaling':
        weights = scale_weights(weights, max_leverage)
    return weights
    
# Calculates the Tangency / Max Sharpe Portfolio
def calc_tan_port_weights(assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame, rf : float, 
                          method='max_sharpe', optimizer='CVXPY', 
                          weights_sum=1, lower_bound=None, upper_bound=None, 
                          max_leverage=1.5, max_leverage_method=None, date=None) -> np.array:
    '''
    Calculates the Tangency / Max Sharpe Portfolio
    assets_exp_rets     : expected return vector
    cov_matrix          : covariance matrix
    rf                  : the risk-free rate of return
    method              : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                          optimization problem. 'geometric' uses the efficient frontier coefficients, 
                          but can only be calculate when there are no additional constraints
    optimizer           : the optimizer to use (CVXPY or SciPy)
    weights_sum         : what the weights must sum up, depends on desired net position
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    max_leverage_method : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    
    # Check validity of method and optimizer
    if method not in ['max_sharpe', 'geometric']:
        raise ValueError(f'Invalid method: {method}')
    if optimizer not in ['CVXPY', 'SciPy']:
        raise ValueError(f'Invalid optimizer: {optimizer}')
    
    init_optimizer, init_max_leverage_method = optimizer, max_leverage_method
    max_leverage_methods = [init_max_leverage_method, 'scaling'] if init_max_leverage_method != 'scaling' else ['scaling']
    optimizers = [init_optimizer, 'SciPy'] if init_optimizer != 'SciPy' else ['SciPy']
    
    for optimizer, max_leverage_method in product(optimizers, max_leverage_methods):
    
        if method == 'max_sharpe' and optimizer == 'CVXPY':
            start_time = time.time()
            w = cvx.Variable(len(assets_exp_rets))
            k = cvx.Variable(nonneg=True)
            variance = cvx.quad_form(w, np.asmatrix(cov_matrix))
            
            cons = [cvx.sum(w) == k * weights_sum, 
                    w @ np.array(assets_exp_rets) - rf == 1,
                    k >= 1e-6,
                    k <= 2e3 * len(assets_exp_rets)]
            if max_leverage_method is not None and max_leverage_method == 'constraint':
                cons += [cvx.norm(w, 1) <= max_leverage * k]
            default_bound = 2e3 if weights_sum != 0 else 1e6
            cons += [w >= lower_bound * k] if lower_bound is not None else [w >= -default_bound * k]
            cons += [w <= upper_bound * k] if upper_bound is not None else [w <= default_bound * k]
                
            prob = cvx.Problem(cvx.Minimize(variance), cons)
            solver_kwargs = {'eps_abs' : 1e-6, 'eps_rel' : 1e-6, 'eps_prim_inf' : 1e-15, 'eps_dual_inf'  : 1e-15, 'max_iter' : 100000}# , 'verbose':True}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver='OSQP', **solver_kwargs)
                    # Attempts 5 warm runs if not optimal
                    solve_attempts = 1
                    while prob.status != 'optimal' and solve_attempts < 5:
                        prob.solve(solver='OSQP', **solver_kwargs)
                        solve_attempts += 1
                    status = prob.status
                    # if solve_attempts > 1 and status == 'optimal':
                    #    print(f'In Tan, used {solve_attempts-1} warm runs on {date.date() if date is not None else date}')
            except:
                # if max_leverage_method == 'scaling':
                #     print(f'RuntimeWarning: In TP, CVXPY reached max_iter on {date.date() if date is not None else date} with max_leverage_method \'{max_leverage_method}\'')
                weights = None
                status = 'failure'
            
            if status == 'optimal':
                if weights_sum != 0 and k.value >= 1e-6:
                    weights = w.value / k.value
                else:
                    weights = w.value
                    max_leverage_method = 'scaling'
                break
            else:
                weights = None
                # prob = cvx.Problem(cvx.Minimize(variance), cons)
                # prob.solve(solver='OSQP', **solver_kwargs, verbose=True)
    
        elif method == 'max_sharpe' and optimizer == 'SciPy':
            num_assets = len(assets_exp_rets)
            tan_port_weights_and_k = np.concatenate([np.repeat(1/num_assets, num_assets), [1]])
            
            def standev(tan_port_weights_and_k):
                return np.sqrt(tan_port_weights_and_k[:-1].T @ np.array(cov_matrix) @ tan_port_weights_and_k[:-1])
            bounds = ((None, None),)*num_assets + ((0, None),)
            cons = [{'type': 'eq', 'fun': lambda x: sum(x[:-1]) - x[-1] * weights_sum}]
            cons.append({'type': 'eq', 'fun': lambda x: x[:-1] @ np.array(assets_exp_rets) - rf - 1 })
            if max_leverage_method is not None and max_leverage_method == 'constraint':
                cons.append({'type': 'ineq', 'fun': lambda x: max_leverage * x[-1] - np.linalg.norm(x[:-1], 1)})
            if lower_bound is not None:
                for i in range(num_assets):
                    def lb_constraint(x, i=i): return -lower_bound  * x[-1] + x[i]
                    cons.append({'type': 'ineq', 'fun': lb_constraint })
            if upper_bound is not None:
                for i in range(num_assets):
                    def ub_constraint(x, i=i): return upper_bound  * x[-1] - x[i]
                    cons.append({'type': 'ineq', 'fun': ub_constraint })
            optimization = minimize(standev, tan_port_weights_and_k, method='SLSQP',
                                    bounds = bounds,
                                    options = {'disp':False, 'ftol': 1e-15, 'maxiter': 1e4} ,
                                    constraints=cons)
            
            if optimization.success:
                # print(max(abs(optimization.x[:-1])))
                # print(optimization.x[-1])
                if weights_sum != 0 and not np.isclose(optimization.x[-1], 0):
                    weights = optimization.x[:-1] / optimization.x[-1]
                else:
                    weights = optimization.x[:-1]
                    max_leverage_method = 'scaling'
                break
            else:
                weights = None
                # print(date)
                # print(optimization)

        elif method == 'geometric':
            A, B, C = get_ef_coefs(assets_exp_rets, cov_matrix)
            tan_port_return = (C-B*rf)/(B-A*rf)
            if max_leverage_method == 'constraint':
                max_leverage_method = 'scaling'
                print('RuntimeWarning: Method \'geometric\' cannot use max_leverage_method \'constraint\'. Defaulting to \'scaling\'')
            weights = calc_mean_var_weights(tan_port_return, assets_exp_rets, cov_matrix, optimizer=optimizer,
                                            lower_bound=lower_bound, upper_bound=upper_bound, 
                                            max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=date)
    # if optimizer != init_optimizer:
    #     print(f'RuntimeWarning: In TP, {init_optimizer} solver failed on {date.date() if date is not None else date}. Defaulting to {optimizer} solver.')
    # if max_leverage_method != init_max_leverage_method and weights is not None:
    #     print(f'RuntimeWarning: In TP, max_leverage_method \'{init_max_leverage_method}\' failed on {date.date() if date is not None else date}. Defaulting to \'{max_leverage_method}\'.')
    weights = correct_extreme_weights(weights, weights_sum=weights_sum, assets_exp_rets=assets_exp_rets, cov_matrix=cov_matrix, 
                                      max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=date)
    return weights

def calc_tan_port_weights_mult_weights_sum(assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame, rf : float, weights_sum_list=[1], tan_method='max_sharpe', 
                                           optimizer='CVXPY', lower_bound=None, upper_bound=None, 
                                           max_leverage=1.5, max_leverage_method=None, date=None) -> np.array:
    '''
    Calculates tan port with multiple weight sums and takes the portfolio among the weight sums with highest Sharpe
    assets_exp_rets     : expected return vector
    cov_matrix          : covariance matrix
    rf                  : the risk-free rate of return
    optimizer           : the optimizer to use (CVXPY or SciPy)
    weights_sum_list    : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                          portfolio with the highest Sharpe
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    max_leverage_method : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    tan_port_weights_weights_sum_df = pd.DataFrame(columns=assets_exp_rets.index)
    for weights_sum in weights_sum_list:
        tan_port_weights = calc_tan_port_weights(assets_exp_rets, cov_matrix, rf, method=tan_method, 
                                                 optimizer=optimizer, weights_sum=weights_sum, lower_bound=lower_bound, upper_bound=upper_bound, 
                                                 max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=date)
        tan_port_weights_weights_sum_df.loc[weights_sum] = tan_port_weights
    
    def sharpe(weights, assets_exp_rets, cov_matrix, rf):
        return (weights.T @ assets_exp_rets - rf) / np.sqrt((weights.T @ cov_matrix @ weights)) * np.sqrt(252)
    weights_sum_sharpes = tan_port_weights_weights_sum_df.apply(sharpe, axis=1, args=(assets_exp_rets, cov_matrix, rf))
    max_sharpe_weights = tan_port_weights_weights_sum_df.iloc[weights_sum_sharpes.argmax()]
    return max_sharpe_weights

def select_EF_portfolio_weights(portfolio_selection : str, assets_exp_rets : pd.Series, cov_matrix : pd.DataFrame, rf : float,
                                weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, 
                                tan_method='max_sharpe', optimizer='CVXPY', date=None) -> np.array:
    '''
    Runs either tangency or MVP portfolio opt
    portfolio selection : either 'tangency' or 'MVP'
    '''
    if portfolio_selection == 'tangency':
        port_weights = calc_tan_port_weights_mult_weights_sum(assets_exp_rets, cov_matrix, rf,
                                                              weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, 
                                                              max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                              tan_method=tan_method, optimizer=optimizer, date=date)
    elif portfolio_selection == 'MVP':
        port_weights = calc_MVP_weights(assets_exp_rets, cov_matrix, lower_bound=lower_bound, upper_bound=upper_bound, 
                                        max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=date)
    else:
        raise ValueError(f'Undefinied portfolio selection method: {portfolio_selection}')
    return port_weights

def estimate_markowitz_parameters(assets_returns_df : pd.DataFrame, cov_assets_returns_df=None, full_upto_assets_returns_df=None,
                                  mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={}) -> (pd.Series, pd.DataFrame):
    '''
    Estimates expected return vector and covariance matrix with input estimator functions
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    cov_assets_returns_df       : data of date x asset used for covariance matrix calculation
    full_upto_assets_returns_df : data of date x asset used in some expected return vector calculations that require
                                  the long-run history
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    '''
    cov_assets_returns_df = assets_returns_df.copy() if cov_assets_returns_df is None else cov_assets_returns_df.copy()
    assets_exp_rets = mean_estimator(assets_returns_df.copy(), full_upto_assets_returns_df=full_upto_assets_returns_df, **mean_est_kwargs)
    cov_matrix = cov_estimator(cov_assets_returns_df.copy(), full_upto_assets_returns_df=full_upto_assets_returns_df, **cov_est_kwargs)
    return assets_exp_rets, cov_matrix

def est_markowitz_params_and_select_weights(assets_returns_df : pd.DataFrame, rf : float, portfolio_selection='tangency', 
                                            cov_assets_returns_df=None, full_upto_assets_returns_df=None,
                                            mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                            weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, 
                                            tan_method='max_sharpe', optimizer='CVXPY', date=None) -> np.array:
    '''
    Estimate markowitz params and select weights
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    rf                          : the risk-free rate of return
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
    cov_assets_returns_df       : data of date x asset used for covariance matrix calculation
    full_upto_assets_returns_df : data of date x asset used in some expected return vector calculations that require
                                  the long-run history
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound         : an array-like to constrain each weight to not be below
    upper_bound         : an array-like to constrain each weight to not be above
    max_leverage        : how many times to lever (1 = no leverage)
    max_leverage_method : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer           : the optimizer to use (CVXPY or SciPy)
    date                : the date the optimization is run for, only used in debugging / warning messages
    '''
    assets_exp_rets, cov_matrix = estimate_markowitz_parameters(assets_returns_df, cov_assets_returns_df=cov_assets_returns_df, full_upto_assets_returns_df=full_upto_assets_returns_df,
                                                                mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs)
    if assets_exp_rets.isnull().any() or cov_matrix.isnull().values.any():
        raise ValueError('assets expected returns or covariance matrix contains NAN')
    port_weights = select_EF_portfolio_weights(portfolio_selection, assets_exp_rets, cov_matrix, rf,
                                               weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                               tan_method=tan_method, optimizer=optimizer, date=date)
    return port_weights

def est_markowitz_params_and_select_weights_in_group(indices : pd.Series, assets_returns_df : pd.DataFrame, rfs : pd.Series, portfolio_selection='tangency', 
                                                     mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                                     weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, 
                                                     tan_method='max_sharpe', optimizer='CVXPY', max_leverage_method=None, lookback=None, cov_lookback=None) -> pd.Series:
    '''
    Estimate markowitz params and select weights when grouped by indices
    indices : indices of the asset_returns_df in the group
    '''
    indices = pd.Series(indices)
    # print(indices.iloc[-1])
    cur_assets_returns_df = assets_returns_df.loc[indices].copy()
    cur_assets_returns_df = cur_assets_returns_df.iloc[-lookback:].copy() if (lookback is not None) else cur_assets_returns_df
    cur_cov_assets_returns_df = cur_assets_returns_df.iloc[-cov_lookback:].copy() if (cov_lookback is not None) else cur_assets_returns_df.copy()
    full_upto_assets_returns_df = assets_returns_df[assets_returns_df.index <= indices.iloc[-1]].copy()
    rf = rfs.loc[indices].iloc[-1].copy()
    
    port_weights = est_markowitz_params_and_select_weights(cur_assets_returns_df, rf, portfolio_selection=portfolio_selection, cov_assets_returns_df=cur_cov_assets_returns_df, 
                                                           full_upto_assets_returns_df=full_upto_assets_returns_df,
                                                           mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                                           weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, 
                                                           tan_method=tan_method, optimizer=optimizer, max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=indices.iloc[-1])
    return pd.Series(port_weights, index=assets_returns_df.columns)

def calc_all_markowitz_weights_daily_rolling(assets_returns_df : pd.DataFrame, lookback : int, rfs : pd.Series, 
                                             portfolio_selection='tangency',
                                             mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                             weights_sum_list=[1], lower_bound=None, upper_bound=None, cov_lookback=None, 
                                             max_leverage=1.5, max_leverage_method=None, tan_method='max_sharpe', optimizer='CVXPY',
                                             ) -> pd.DataFrame:
    '''
    Calculate mean-var weights with a rolling window updating daily
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    lookback                    : number of days in the lookback window
    rfs                         : a series of the risk-free rate of return for each day
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
    cov_assets_returns_df       : data of date x asset used for covariance matrix calculation
    full_upto_assets_returns_df : data of date x asset used in some expected return vector calculations that require
                                  the long-run history
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    '''
    max_lookback = max(lookback, cov_lookback) if cov_lookback is not None else lookback
    cov_lookback = cov_lookback if cov_lookback is not None else lookback

    port_weights_df = pd.DataFrame(rolling_apply(est_markowitz_params_and_select_weights_in_group, max_lookback, assets_returns_df.index, 
                                                 assets_returns_df=assets_returns_df, rfs=rfs, portfolio_selection=portfolio_selection,
                                                 mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                                 weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                                 max_leverage_method=max_leverage_method, lookback=lookback, cov_lookback=cov_lookback, tan_method=tan_method, optimizer=optimizer), 
                                   columns=assets_returns_df.columns, index=assets_returns_df.index)
    port_weights_df = port_weights_df.iloc[max_lookback:]
    return port_weights_df

def calc_all_markowitz_weights_daily_expanding(assets_returns_df : pd.DataFrame, min_lookback : int, rfs : pd.Series, 
                                               portfolio_selection='tangency',
                                               mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                               weights_sum_list=[1], lower_bound=None, upper_bound=None, min_cov_lookback=None, 
                                               max_leverage=1.5, max_leverage_method=None, tan_method='max_sharpe', optimizer='CVXPY',
                                              ) -> pd.DataFrame:
    '''
    Calculate mean-var weights with a rolling window updating daily
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    min_lookback                : minimum number of days in the expanding window able to calculate for
    rfs                         : a series of the risk-free rate of return for each day
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
    cov_assets_returns_df       : data of date x asset used for covariance matrix calculation
    full_upto_assets_returns_df : data of date x asset used in some expected return vector calculations that require
                                  the long-run history
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    '''
    min_lookback = max(min_lookback, min_cov_lookback) if min_cov_lookback is not None else min_lookback
    min_cov_lookback = min_cov_lookback if min_cov_lookback is not None else min_lookback

    port_weights_df = pd.DataFrame(expanding_apply(est_markowitz_params_and_select_weights_in_group, min_lookback, assets_returns_df.index, 
                                                   assets_returns_df=assets_returns_df, rfs=rfs, portfolio_selection=portfolio_selection,
                                                   mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                                   weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                                   max_leverage_method=max_leverage_method, lookback=None, tan_method=tan_method, optimizer=optimizer), 
                                   columns=assets_returns_df.columns, index=assets_returns_df.index)
    port_weights_df = port_weights_df.iloc[min_lookback:]
    return port_weights_df
    
def calc_all_markowitz_weights_yearly(assets_returns_df : pd.DataFrame, rfs : pd.Series, 
                                      portfolio_selection='tangency', start_date=None,
                                      mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                      weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None,
                                      tan_method='max_sharpe', optimizer='CVXPY') -> pd.DataFrame:
    '''
    Calculate mean-var weights yearly
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    rfs                         : a series of the risk-free rate of return for each day
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
                                  the long-run history
    start_date                  : the date to start running from
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    '''
    dates_df = pd.DataFrame({'Date' : assets_returns_df.index, 'Year' : assets_returns_df.index.year})
    dates_df = dates_df[dates_df['Date'] >= start_date] if start_date is not None else dates_df
    dates_df = dates_df[dates_df['Year'] < dates_df['Year'].iloc[-1]]
    prior_dates_df = pd.DataFrame({'Date' : assets_returns_df.index, 'Last_Year' : assets_returns_df.index.year-1})
    
    yearly_port_weights_df = (dates_df.groupby('Year')['Date']
                              .apply(est_markowitz_params_and_select_weights_in_group,
                                     assets_returns_df=assets_returns_df, rfs=rfs, portfolio_selection=portfolio_selection, 
                                     mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                     weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                     max_leverage_method=max_leverage_method, tan_method=tan_method, optimizer=optimizer)).unstack(level=1)

    port_weights_df = yearly_port_weights_df.copy().merge(prior_dates_df, left_index=True, right_on='Last_Year')
    port_weights_df.index = port_weights_df['Date'].values
    port_weights_df = port_weights_df.drop(columns=['Date', 'Last_Year'])
    return port_weights_df

def calc_all_markowitz_weights_quarterly(assets_returns_df, rfs, portfolio_selection='tangency', start_date=None,
                                         mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                         weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None,
                                         tan_method='max_sharpe', optimizer='CVXPY') -> pd.DataFrame:
    '''
    Calculate mean-var weights quarterly
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    rfs                         : a series of the risk-free rate of return for each day
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
                                  the long-run history
    start_date                  : the date to start running from
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    '''
    dates_df = pd.DataFrame({'Date' : assets_returns_df.index, 'Quarter' : get_quarters(assets_returns_df.index)})
    dates_df = dates_df[dates_df['Date'] >= start_date] if start_date is not None else dates_df
    prior_dates_df = pd.DataFrame({'Date' : dates_df['Date'], 'Last_Quarter' : [get_last_quarter(quarter) for quarter in dates_df['Quarter']] })
    dates_df = dates_df[dates_df['Quarter'] < dates_df['Quarter'].iloc[-1]]
    
    quarterly_port_weights_df = (dates_df.groupby('Quarter')['Date']
                                 .apply(est_markowitz_params_and_select_weights_in_group,
                                        assets_returns_df=assets_returns_df, rfs=rfs, portfolio_selection=portfolio_selection, 
                                        mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                        weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                        max_leverage_method=max_leverage_method, tan_method=tan_method, optimizer=optimizer)).unstack(level=1)
    
    port_weights_df = quarterly_port_weights_df.merge(prior_dates_df, left_index=True, right_on='Last_Quarter')
    port_weights_df.index = port_weights_df['Date'].values
    port_weights_df = port_weights_df.drop(columns=['Date', 'Last_Quarter'])
    return port_weights_df

def calc_all_markowitz_weights_monthly(assets_returns_df, rfs, portfolio_selection='tangency', start_date=None,
                                       mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                       weights_sum_list=[1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None,
                                       tan_method='max_sharpe', optimizer='CVXPY') -> pd.DataFrame:
    '''
    Calculate mean-var weights monthly
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    rfs                         : a series of the risk-free rate of return for each day
    portfolio_selection         : the portfolio selection method ('tangency' or 'MVP')
                                  the long-run history
    start_date                  : the date to start running from
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    weights_sum_list            : a list of different weights sums. Calculates tangency portfolio for each, then takes  the
                                  portfolio with the highest Sharpe
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    '''
    month_first_dates = [dt.datetime(date.year, date.month, 1) for date in assets_returns_df.index]
    dates_df = pd.DataFrame({'Date' : assets_returns_df.index, 'Month' : month_first_dates})
    dates_df = dates_df[dates_df['Date'] >= start_date] if start_date is not None else dates_df
    dates_df = dates_df[dates_df['Month'] < dates_df['Month'].iloc[-1]]
    last_month_first_test_dates = [dt.datetime(date.year-1 if date.month == 1 else date.year, date.month-1 if date.month > 1 else 12, 1) for date in assets_returns_df.index]
    prior_dates_df = pd.DataFrame({'Date' : assets_returns_df.index, 'Last_Month' : last_month_first_test_dates})
    
    monthly_port_weights_df = (dates_df.groupby('Month')['Date']
                               .apply(est_markowitz_params_and_select_weights_in_group,
                                      assets_returns_df=assets_returns_df, rfs=rfs, portfolio_selection=portfolio_selection,
                                      mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                      weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                      max_leverage_method=max_leverage_method, tan_method=tan_method, optimizer=optimizer)).unstack(level=1)
    
    port_weights_df = monthly_port_weights_df.merge(prior_dates_df, left_index=True, right_on='Last_Month')
    port_weights_df.index = port_weights_df['Date'].values
    port_weights_df = port_weights_df.drop(columns=['Date', 'Last_Month'])
    return port_weights_df

def calc_all_ef_coefs(lookback_method : str, assets_returns_df : pd.DataFrame, lookback=None, cov_lookback=None, 
                      mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={}) -> pd.DataFrame:
    '''
    Calculate all the efficient frontier coefs for rolling, yearly, quarterly, or monthly
    lookback_method             : 'rolling', 'yearly', 'quarterly', or 'monthly'
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    lookback                    : number of days in the expected returns lookback window
    cov_lookback                : number of days in the covariance lookback window
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    '''
    coefs_df = pd.DataFrame(columns=['A', 'B', 'C'])
    lookback = init_lookback_period(lookback_method, lookback)
    max_lookback = max(lookback, cov_lookback) if cov_lookback is not None else lookback
    start_date = assets_returns_df.index[max_lookback-1]
    test_dates = assets_returns_df.index[assets_returns_df.index >= start_date]
    if(lookback_method == 'rolling'):
        for cur_date in test_dates:
            # print(cur_date)
            if(lookback is None):
                raise ValueError('The lookback_period is not specified')
            cur_assets_ret_df = assets_returns_df.loc[assets_returns_df.index <= cur_date].iloc[-lookback:].copy()
            cur_cov_assets_ret_df = assets_returns_df.loc[assets_returns_df.index <= cur_date].iloc[-cov_lookback:].copy() if (cov_lookback is not None) else cur_assets_ret_df
            return_vec = mean_estimator(cur_assets_ret_df)
            cov_matrix = cov_estimator(cur_cov_assets_ret_df)
            coefs_df.loc[cur_date] = get_ef_coefs(return_vec, cov_matrix)
    elif(lookback_method == 'yearly'):
        for year in assets_returns_df.index.year.unique().values:
            # print(year)
            cur_assets_ret_df = assets_returns_df.loc[assets_returns_df.index.year == year].copy()
            cur_cov_assets_ret_df = assets_returns_df.loc[assets_returns_df.index <= year].iloc[-cov_lookback:].copy() if (cov_lookback is not None) else cur_assets_ret_df
            return_vec = mean_estimator(cur_assets_ret_df)
            cov_matrix = cov_estimator(cur_cov_assets_ret_df)
            coefs_df.loc[year] = get_ef_coefs(return_vec, cov_matrix)
    elif(lookback_method == 'quarterly'):
        assets_ret_df_quarters = assets_returns_df.copy()
        assets_ret_df_quarters.index = get_quarters(assets_ret_df_quarters.index)
        for quarter in assets_ret_df_quarters.index.unique().values:
            # print(quarter)
            cur_assets_ret_df = assets_ret_df_quarters.loc[assets_ret_df_quarters.index == quarter].copy()
            cur_cov_assets_ret_df = assets_returns_df.loc[assets_returns_df.index <= quarter].iloc[-cov_lookback:].copy() if (cov_lookback is not None) else cur_assets_ret_df
            return_vec = mean_estimator(cur_assets_ret_df)
            cov_matrix = cov_estimator(cur_cov_assets_ret_df)
            coefs_df.loc[quarter] = get_ef_coefs(return_vec, cov_matrix)
    elif(lookback_method == 'monthly'):
        month_first_dates = [dt.datetime(date.year, date.month, 1) for date in assets_returns_df.index]
        for month_first_date in np.unique(month_first_dates):
            # print(month_first_date)
            cur_assets_ret_df = assets_returns_df.loc[(assets_returns_df.index.year == month_first_date.year) & (assets_returns_df.index.month == month_first_date.month)].copy()
            cur_cov_assets_ret_df = assets_returns_df.loc[assets_returns_df.index <= month_first_date].iloc[-cov_lookback:].copy() if (cov_lookback is not None) else cur_assets_ret_df
            return_vec = mean_estimator(cur_assets_ret_df)
            cov_matrix = cov_estimator(cur_cov_assets_ret_df)
            coefs_df.loc[month_first_date] = get_ef_coefs(return_vec, cov_matrix)
    else:
        raise ValueError(lookback_method+' is not a valid lookback_method')    
    coefs_df['r_MVP'] = coefs_df['B']/coefs_df['A']
    coefs_df['sigma_MVP'] = 1/np.sqrt(coefs_df['A'])
    coefs_df['u'] = np.sqrt(np.maximum((coefs_df['A']*coefs_df['C'] - coefs_df['B']**2)/coefs_df['A'], 0))
    for ef_coef in ['r_MVP', 'sigma_MVP', 'u']:
        coefs_df[ef_coef+'_diff'] = coefs_df[ef_coef] - coefs_df[ef_coef].shift(1)
    return coefs_df

# Calculate all the mean-var weights and EF coefs for rolling, yearly, quarterly, monthly
def calc_all_markowitz_weights_and_coefs(portfolio_name : str, asset_set : str, subfolder_name : str, assets_ret_df : pd.DataFrame, rfs : pd.Series,
                                         portfolio_selection='tangency', lookback_method='rolling', lookback=None, cov_lookback=None, 
                                         mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={},
                                         lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, 
                                         tan_method='max_sharpe', optimizer='CVXPY',
                                         find_weights=True, find_coefs=True, save_folder='.') -> None:
    '''
    Calculate all the efficient frontier coefs for rolling, yearly, quarterly, or monthly
    portfolio_name              : the portfolio name when saving the filename
    asset_set                   : the name of the universe of assets
    subfolder_name              : the subfolder version name
    assets_returns_df           : data of date x asset used for expected returns vector calculation
    rfs                         : a series of risk-free returns for each day
    portfolio_selection         : 'tangency' or 'MVP'
    lookback_method             : 'rolling', 'yearly', 'quarterly', or 'monthly'
    lookback                    : number of days in the expected returns lookback window
    cov_lookback                : number of days in the covariance lookback window
    mean_estimator              : a callable used to estimate the expected return vector
    cov_estimator               : a callable used to estimate the covariance matrix
    mean_est_kwargs             : kwargs for the mean estimator
    cov_est_kwargs              : cov_estimator kwargs
    lower_bound                 : an array-like to constrain each weight to not be below
    upper_bound                 : an array-like to constrain each weight to not be above
    max_leverage                : how many times to lever (1 = no leverage)
    max_leverage_method         : how to limit max_leverage. Will try 'constraint' first then default to 'scaling'
    tan_method                  : 'max_sharpe' maximizes the Sharpe objective function reformulated as a conic
                                  optimization problem. 'geometric' uses the efficient frontier coefficients, 
                                  but can only be calculate when there are no additional constraints 
    optimizer                   : the optimizer to use (CVXPY or SciPy)
    find_weights                : whether to calculate the weights
    find_coefs                  : whethter to calculate the efficient frontier coefficients
    save_folder                 : the directory to save results under
    '''
    if(find_weights):
        if(lookback_method == 'rolling'):
            port_weights_df = calc_all_markowitz_weights_daily_rolling(assets_ret_df, lookback, rfs, portfolio_selection=portfolio_selection, 
                                                                       cov_lookback=cov_lookback, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                                       lower_bound=lower_bound, upper_bound=upper_bound, tan_method=tan_method, optimizer=optimizer)
        elif(lookback_method == 'yearly'):
            port_weights_df = calc_all_markowitz_weights_yearly(assets_ret_df, rfs, portfolio_selection=portfolio_selection,
                                                                max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                                lower_bound=lower_bound, upper_bound=upper_bound, tan_method=tan_method, optimizer=optimizer)
        elif(lookback_method == 'quarterly'):
            port_weights_df = calc_all_markowitz_weights_quarterly(assets_ret_df, rfs, portfolio_selection=portfolio_selection,
                                                                   max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                                   lower_bound=lower_bound, upper_bound=upper_bound, tan_method=tan_method, optimizer=optimizer)
        elif(lookback_method == 'monthly'):
            port_weights_df = calc_all_markowitz_weights_monthly(assets_ret_df, rfs, portfolio_selection=portfolio_selection,
                                                                 max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
                                                                 lower_bound=lower_bound, upper_bound=upper_bound, tan_method=tan_method, optimizer=optimizer)
        else:
            raise ValueError(lookback_method+' is not a valid lookback_method')
        subfolder_dir = '../Portfolios/'+asset_set+'/Versions/'+subfolder_name+'/Weights/'+save_folder+'/'
        if not os.path.exists(subfolder_dir):
            os.makedirs(subfolder_dir)
        port_weights_df.to_csv(subfolder_dir+'/daily_weights_'+portfolio_name+'.csv')
        print(f'Finish calculating weights for {portfolio_name}. Abs Max:', round(port_weights_df.abs().sum(axis=1).max(), 2))
    if(find_coefs):
        if(lookback_method == 'rolling'):
            coefs_df = calc_all_ef_coefs('rolling', assets_ret_df, lookback=lookback, cov_lookback=cov_lookback)
        elif(lookback_method == 'yearly'):
            coefs_df = calc_all_ef_coefs('yearly', assets_ret_df)
        elif(lookback_method == 'quarterly'):
            coefs_df = calc_all_ef_coefs('quarterly', assets_ret_df)
        elif(lookback_method == 'monthly'):
            coefs_df = calc_all_ef_coefs('monthly', assets_ret_df)
        else:
            raise ValueError(f'Invalid lookback method: {lookback_method}')
        ef_coefs_dir = '../Portfolios/'+asset_set+'/EF_Coefs/'+save_folder+'/'
        if not os.path.exists(ef_coefs_dir):
            os.makedirs(ef_coefs_dir)
        coefs_df.to_csv(ef_coefs_dir+'EF_coefs_'+portfolio_name[4:]+'.csv')

