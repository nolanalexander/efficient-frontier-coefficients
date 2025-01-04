import numpy as np
import pandas as pd
from scipy.stats import norm

'''
Calculations of Black-Scholes, Greeks, and Shadow Greeks based on https://www.macroption.com/black-scholes-formula/
'''

exp = np.exp
ln = np.log
n = norm.pdf
N = norm.cdf
N_inv = norm.ppf
sqrt = np.sqrt

def func_call_put(call_put, call_func, put_func):
    if call_put in ('Call', 'call', 'C', 'c'):
        return call_func()
    elif call_put in ('Put', 'put', 'P', 'p'):
        return put_func()
    else:
        raise ValueError(f'Invalid call_put: {call_put}')

def d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration):
    S = underlying_price
    K = strike_price
    v = volatility
    r = risk_free_rate
    q = dividend_yield
    t = time_to_expiration
    
    n1 = ln(S / K)
    n2 = t * (r - q + ((v ** 2) / 2))
    d = v * sqrt(t)
    return (n1 + n2) / d

def d2(volatility, time_to_expiration, d1):
    v = volatility
    t = time_to_expiration
    
    return d1 - v * sqrt(t)

# Rate of change of option price relative to change in underling price
def delta(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=None):
    q = dividend_yield
    t = time_to_expiration
    d1 = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    
    z = exp(-q * t)
    call_func = (lambda: z *  N(d1))
    put_func  = (lambda: z * (N(d1) - 1))
    return func_call_put(call_put, call_func, put_func)

# Rate of change of option price relative to change in volatility
def vega(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=None):
    S = underlying_price
    q = dividend_yield
    t = time_to_expiration
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    
    return S * exp(-q * t) * sqrt(t) * n(d1_) / 100

# Rate of change of Delta relative to change in underlying price
def gamma(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=None):
    S = underlying_price
    q = dividend_yield
    t = time_to_expiration
    v = volatility
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    
    return (exp(-q * t) / (S * v * sqrt(t))) * n(d1_)

# Rate of change of Delta relative to change in underlying price
def vanna(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=None, vega=None):
    S = underlying_price
    t = time_to_expiration
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    vega_ = vega(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1) if vega is None else vega
    
    return (vega_ / S) * (1 - (d1_ / (volatility * sqrt(t))))

# Rate of change of Vega relative to change in volatility
def vomma(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=None, d2=None, vega=None):
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    d2_ = d2(volatility, time_to_expiration, d1) if d2 == d2 else d2
    vega_ = vega(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1) if vega is None else vega
    
    return vega_ * ((d1_ * d2_) / volatility) / 100

# Rate of change of option price relative to change in risk-free rate
def rho(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=None, d2=None):
    K = strike_price
    r = risk_free_rate
    t = time_to_expiration
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    d2_ = d2(volatility, time_to_expiration, d1) if d2 == d2 else d2
    
    z = K * t * exp(-r * t) / 100
    call_func = (lambda: z * N( d1_))
    put_func  = (lambda: z * N(-d2_))
    return func_call_put(call_put, call_func, put_func)

# Rate of decay of option due to time
def theta(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=None, d2=None):
    S = underlying_price
    K = strike_price
    r = risk_free_rate
    q = dividend_yield
    t = time_to_expiration
    days_per_year = 252
    v = volatility
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    d2_ = d2(volatility, time_to_expiration, d1_) if d2 is None else d2
    
    t1 = 1 / days_per_year
    t2 = ((S * v * exp(-q * t)) / (2 * sqrt(t))) * n(d1)
    t3 = r * K * exp(-r * t)
    t4 = q * S * exp(-q * t)
    call_func = (lambda: t1 * (-t2 - (t3 * N( d2_)) + ( t4 * N( d1_))))
    put_func  = (lambda: t1 * (-t2 + (t3 * N(-d2_)) - ( t4 * N(-d1_))))
    return func_call_put(call_put=call_put, call_func=call_func, put_func=put_func)

# Black-Scholes pricing
def option_price(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=None, d2=None):
    S = underlying_price
    K = strike_price
    r = risk_free_rate
    q = dividend_yield
    t = time_to_expiration
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration) if d1 is None else d1
    d2_ = d2(volatility, time_to_expiration, d1) if d2 is None else d2
    
    call_func = (lambda: S * exp(-q * t) * N( d1_) - K * exp(-r * t) * N( d2_))
    put_func =  (lambda: K * exp(-r * t) * N(-d2_) - S * exp(-q * t) * N(-d1_))
    return func_call_put(call_put=call_put, call_func=call_func, put_func=put_func)

# Implied volatility calculated using Newton's method
def implied_volatility(option_price, underlying_price, strike_price, risk_free_rate, dividend_yield, time_to_expiration, call_put,
                       init_guess=0.5, min_volatility=1e-15, max_volatility=1e15, precision=None, bisect_ratio=0.9, max_iter=100):
    precision = precision if precision is not None else strike_price * 1e-10
    max_iter = max_iter if max_iter is not None else 100
    
    min_price = option_price(underlying_price, strike_price, min_volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put)
    is_valid_price = option_price >= min_price
    
    for i in range(1, max_iter + 1):
        cur_option_price = option_price(underlying_price, strike_price, cur_volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put)
        price_diff = cur_option_price - option_price
        
        is_tolerable = np.abs(cur_option_price - option_price) <= precision
        is_nan = pd.isnull(cur_option_price)
        is_done = is_tolerable | is_nan | ~is_valid_price
        all_done = is_done if type(is_done) is np.bool_ else all(is_done)
        
        if all_done or i >= max_iter:
            volatility = np.where(is_tolerable & is_valid_price, cur_volatility, np.nan)
            return volatility
        
        min_volatility = np.where(price_diff <= 0, np.maximum(cur_volatility, min_volatility), min_volatility)
        max_volatility = np.where(price_diff >= 0, np.minimum(cur_volatility, max_volatility), max_volatility)
        
        cur_vega = vega(underlying_price, strike_price, cur_volatility, risk_free_rate, dividend_yield, time_to_expiration)
        volatility_step = -0.01 * (price_diff / cur_vega)
        cur_volatility_newton = cur_volatility + volatility_step
        newton_in_bounds= ((cur_volatility_newton >= min_volatility) & 
                           (cur_volatility_newton <= max_volatility))
        cur_volatility_bisect = ((min_volatility * bisect_ratio) +
                                 (max_volatility * (1 - bisect_ratio)))
        cur_volatility = np.where(newton_in_bounds, cur_volatility_newton, cur_volatility_bisect)
        
    raise RuntimeError(f'IV was not returned after {max_iter} attempts')

def strike_from_delta(delta, volatility, underlying_price, risk_free_rate, dividend_yield, time_to_expiration, call_put):
    S = underlying_price
    v = volatility
    r = risk_free_rate
    q = dividend_yield
    t = time_to_expiration
    
    z = exp(-q * t)
    d1_call_func = (lambda: N_inv( delta / z))
    d1_put_func  = (lambda: N_inv((delta / z) + 1))
    d1_ = func_call_put(call_put, d1_call_func, d1_put_func)
    n2 = t * (r - q + ((v ** 2) / 2))
    d = v * sqrt(t)
    return S / exp((d1_ * d) - n2)

def calc_iv_and_greeks(option_price, underlying_price, strike_price, risk_free_rate, dividend_yield, time_to_expiration, call_put):
    iv_and_greeks = pd.Series(dtype=float)
    iv_and_greeks.loc['Implied Volatility'] = implied_volatility(option_price, underlying_price, strike_price, risk_free_rate, dividend_yield, time_to_expiration, call_put)
    volatility = bs_and_greeks.loc['Implied Volatility']
    d1_ = d1(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration)
    d2_ = d2(volatility, time_to_expiration, d1)
    iv_and_greeks.loc['Delta'] = delta(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=d1_)
    iv_and_greeks.loc['Vega']  = vega(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1_)
    iv_and_greeks.loc['Gamma'] = gamma(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1_)
    iv_and_greeks.loc['Vanna'] = vanna(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1_, vega=iv_and_greeks.loc['Vega'])
    iv_and_greeks.loc['Vomma'] = vomma(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, d1=d1_, d2=d2_, vega=iv_and_greeks.loc['Vega'])
    iv_and_greeks.loc['Rho']   = rho(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=d1_, d2=d2_)
    iv_and_greeks.loc['Theta'] = theta(underlying_price, strike_price, volatility, risk_free_rate, dividend_yield, time_to_expiration, call_put, d1=d1_, d2=d2_)
    return iv_and_greeks
    
    
    
    
    
    
    
    
    
    
    
