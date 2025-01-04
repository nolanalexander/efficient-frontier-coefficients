import pandas as pd
import numpy  as np

import options.black_scholes_greeks as bs

call_price         = 5.83
put_price          = 7.74
strike_price       = 121
underlying_price   = 119.1832341593
risk_free_rate     = 0.01
dividend_yield     = 0.01
close_date         = '2022-06-10'
expiration_date    = '2022-07-15'
time_to_expiration = np.busday_count(close_date, expiration_date) / 252


call_iv_and_greeks = bs.calc_iv_and_greeks(call_price, underlying_price, strike_price, risk_free_rate, dividend_yield, time_to_expiration, 'Call')
put_iv_and_greeks = bs.calc_iv_and_greeks(put_price, underlying_price, strike_price, risk_free_rate, dividend_yield, time_to_expiration, 'Put')
print(call_iv_and_greeks)
print(put_iv_and_greeks)

bs_call_price = bs.option_price(underlying_price, strike_price, call_iv_and_greeks.loc['Implied Volatility'], risk_free_rate, dividend_yield, time_to_expiration, 'Call')
bs_put_price = bs.option_price(underlying_price, strike_price, put_iv_and_greeks.loc['Implied Volatility'], risk_free_rate, dividend_yield, time_to_expiration, 'Put')
print(bs_call_price)
print(bs_put_price)





