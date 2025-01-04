import pandas as pd
import numpy as np
from pyfolio import timeseries
import pyfolio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")

from models.reinforcement_learning.StockTradingEnv import StockTradingEnv
from models.reinforcement_learning.DRLAgent import DRLAgent
import time

def get_rl_backtest_actions(train_df, test_df, predictors, epochs=50, ent_coef=0.01, learning_rate=1e-4, days_fwd=1):

    start_time = time.time()
    
    # calculate state action space
    stock_dimension = len(train_df['Ticker'].unique())
    state_space = (1 + 2 * stock_dimension + len(predictors) * stock_dimension)
    
    env_kwargs = {
        "initial_amount": 1, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": predictors, 
        "action_space": stock_dimension, 
        "reward_scaling": 1
        }
    
    e_train_gym = StockTradingEnv(df=train_df[predictors + ['Ticker', 'Date', 'Chg_'+str(days_fwd)+'D_Fwd']], **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    
    print("==============Model Training===========")
    
    ppo_params = {
        "n_steps": 2048,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "batch_size": 64,
    }
    
    model_ppo = agent.get_model("ppo", model_kwargs=ppo_params)
    trained_ppo = agent.train_model(
        model=model_ppo, tb_log_name="ppo", total_timesteps=epochs*len(train_df['Date'].unique())
    )
    
    print("==============Start Trading===========")
    e_trade_gym = StockTradingEnv(df=test_df[predictors + ['Ticker', 'Date', 'Chg_'+str(days_fwd)+'D_Fwd']], **env_kwargs)
    
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_ppo, environment = e_trade_gym
    )
    
    print('DRL Runtime: ' + str((time.time() - start_time)/60))
    
    return df_actions, df_account_value

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret= df.copy()
    strategy_ret['date'] = pd.to_datetime(strategy_ret['date'])
    strategy_ret.set_index('date', drop = False, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts

def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def backtest_plot(
    account_value,
    value_col_name="account_value",
):

    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, set_context=False
        )

def trx_plot(df_trade,df_actions,ticker_list):    
    df_trx = pd.DataFrame(np.array(df_actions['transactions'].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions['date']
    df_trx.index.name = ''
    
    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:,i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: True if x>0 else False)
        selling_signal = df_trx_temp_sign.apply(lambda x: True if x<0 else False)
        
        tic_plot = df_trade[(df_trade['tic']==df_trx_temp.name) & (df_trade['date'].isin(df_trx.index))]['close']
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize = (10, 8))
        plt.plot(tic_plot, color='g', lw=2.)
        plt.plot(tic_plot, '^', markersize=10, color='m', label = 'buying signal', markevery = buying_signal)
        plt.plot(tic_plot, 'v', markersize=10, color='k', label = 'selling signal', markevery = selling_signal)
        plt.title(f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal==True]) + len(selling_signal[selling_signal==True])}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25)) 
        plt.xticks(rotation=45, ha='right')
        plt.show()
