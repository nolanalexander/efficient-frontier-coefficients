import pandas as pd
import numpy as np
import matplotlib
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

import config as config
from AssetTradingEnv import AssetTradingEnv
from DRLAgent import DRLAgent
from rl_backtest import backtest_stats
import datetime as dt

import itertools
import time

start_time = time.time()

# Read in and Preprocess Data
assets_set = 'SPDR_ETF_Sectors'
assets_df_wmiss = pd.read_csv('../../Portfolios/'+assets_set+'Assets_Data/assets_data.csv')
assets_df_wmiss['Date'] = pd.to_datetime(assets_df_wmiss['Date'])
assets_df_wmiss['Return_Binary'] =  ["Up" if x > 0 else "Down" for x in assets_df_wmiss["Chg_Next"]]
tickers_with_inf = assets_df_wmiss['Ticker'][np.isinf(assets_df_wmiss['A/D_Osci'])].unique() # days that had the same high and low
assets_df_wmiss = assets_df_wmiss[~assets_df_wmiss['Ticker'].isin(tickers_with_inf)]

def preprocess_assets_df(assets_df_wmiss):
    assets_df = assets_df_wmiss.dropna()
    row_is_last_date = (assets_df['Date'] == max(assets_df['Date']))
    if(sum(row_is_last_date) != len(assets_df['Date'].unique())):
        assets_df = assets_df[~row_is_last_date]
    assets_df = assets_df.reset_index(drop=True)
    assets_df['Date'] = pd.to_datetime(assets_df['Date'])
    return assets_df

def get_returns_df(assets_df):
    returns_df = assets_df.set_index(['Date', 'Ticker'])[['Chg_Next']].unstack(level=-1).copy()
    returns_df.columns = returns_df.columns.levels[1]
    return returns_df

def calc_strategy_returns(actions_df, returns_df, save_csv=True):
    strategy_returns = pd.Series(dtype=float)
    for date in actions_df.index.values:
        strategy_returns.loc[date] = sum(actions_df.loc[date] * returns_df.loc[date])
    if(save_csv):
        strategy_returns.to_csv('strategy_returns.csv')
    return strategy_returns
    
def calc_metrics_by_year(strategy_returns, actions_df, save_csv=True):
    years = strategy_returns.index.to_series().dt.year.unique()
    metrics = pd.DataFrame(columns=['Sharpe', 'Mean_Return', 'Median_Return', 'SD_Return', 'Min_Return', 'Max_Return', 
                                    'Percent_Long', 'Percent_Short', 'Percent_No_Trade'])
    def calc_metrics(strategy_returns, actions_df):
        action_counts = pd.Series(actions_df.values.ravel()).dropna().value_counts()
        return [np.sqrt(252) * strategy_returns.mean()/strategy_returns.std(), strategy_returns.mean(), strategy_returns.median(),
                strategy_returns.std(), strategy_returns.min(), strategy_returns.max(), 
                action_counts[1]/action_counts.sum(), action_counts[-1]/action_counts.sum(), action_counts[0]/action_counts.sum()]
    metrics.loc['Full'] = calc_metrics(strategy_returns, actions_df)
    for year in years:
        metrics.loc[str(year)] = calc_metrics(strategy_returns[strategy_returns.index.to_series().dt.year == year], 
                                              actions_df[actions_df.index.to_series().dt.year == year])
    if(save_csv):
        metrics.to_csv('strategy_metrics.csv')
    return metrics

assets_df = preprocess_assets_df(assets_df_wmiss)

# Predictors
base_predictors = np.array(['Chg', 'Chg_1lag', 'Chg_2to5lag', 'SPX_Chg', 'SPX_Chg_1lag', 'SPX_Chg_2to5lag', 
                            'VIX', '10D_Vol / 60D_Vol', '22D_Vol / VIX', 'SPX 10D_Vol / SPX 60D_Vol',
                            'SPX 5D_Chg / 20D_Chg', 'SPX_Close / SPX_22D_Close_MA', 'Close / 22D_Close_MA', 
                            'Beta_SPX'])
cluster_predictors = np.array(['Cohort_Chg', 'Cohort_Chg_1lag', 'Cohort_Chg_2to5lag', 'Cohort_Beta'])
technical_predictors = np.array(['Sto_K', 'Sto_D', 'Slow_Sto_D', 'Momentum', 'ROC', 'Williams_R', 'A/D_Osci',
                                 'Disparity_5', 'Disparity_10', 'OSCP', 'CCI', 'RSI'])
base_wo_spx_predictors = np.array(['Chg', 'Chg_1lag', 'Chg_2to5lag', '10D_Vol / 60D_Vol', '22D_Vol / VIX',
                                             'Close / 22D_Close_MA', 'Beta_SPX'])
base_wo_spx_technical_predictors = np.append(base_wo_spx_predictors, technical_predictors)
new_predictors = np.array(['Sto_K', 'Chg_Slope5D', 'Chg_Slope5D_Slope5D']) # , 'Winrate_22D', 'Chg_Chg'
base_cluster_predictors = np.append(base_predictors, cluster_predictors)
base_technical_predictors = np.append(base_predictors, technical_predictors)
minimal_predictors = np.array(['Chg', 'Chg_1lag', 'Chg_2lag', 'Chg_3lag', 'Chg_4lag', 'Chg_5lag', 
                               '22D_Vol / VIX', 'Beta_SPX', 'Chg_Slope5D', 'Chg_Slope5D_Slope5D'])

assets_df = assets_df[np.append(['Date','Ticker', 'Chg_Next'], base_wo_spx_technical_predictors)]
dates = assets_df['Date'].unique()

test_start_date = dt.datetime(2018, 1, 2)

tech_indicator_list = base_wo_spx_technical_predictors # assets_df.columns[~assets_df.columns.isin(['Date', 'Ticker', 'Chg_Next'])]

list_ticker = assets_df["Ticker"].unique().tolist()
list_date = list(pd.date_range(assets_df['Date'].min(),assets_df['Date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))
comb_df = pd.DataFrame(combination, columns=["Date","Ticker"])
comb_df['Date'] = pd.to_datetime(comb_df['Date'])

processed_full = comb_df.merge(assets_df,on=["Date","Ticker"],how="left")
processed_full = processed_full[processed_full['Date'].isin(assets_df['Date'])]
processed_full = processed_full.sort_values(['Date','Ticker'])
processed_full = processed_full.fillna(0)

# Training & Trading data split
train_df = processed_full[processed_full['Date'] < test_start_date]
test_df = processed_full[processed_full['Date'] >= test_start_date]

# Creates days for index
train_df.index = ((train_df['Date'] - train_df['Date'].iloc[0]) / np.timedelta64(1, 'D')).astype('int')
days = train_df.index.unique().tolist()
train_df.index = [days.index(day) for day in train_df.index]
train_df.index.rename('Day', inplace=True)

test_df.index = ((test_df['Date'] - test_df['Date'].iloc[0]) / np.timedelta64(1, 'D')).astype('int')
days = test_df.index.unique().tolist()
test_df.index = [days.index(day) for day in test_df.index]
test_df.index.rename('Day', inplace=True)

# calculate state action space
stock_dimension = len(list_ticker)
state_space = (1 + 2 * stock_dimension + len(tech_indicator_list) * stock_dimension)

env_kwargs = {
    "initial_amount": 1, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": tech_indicator_list, 
    "action_space": stock_dimension, 
    "reward_scaling": 1 # 1e-3
    }

e_train_gym = AssetTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

agent = DRLAgent(env=env_train)

print("==============Model Training===========")
# now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

model_ppo = agent.get_model("ppo")
trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps= 50*len(train_df['Date'].unique())
)

print("==============Start Trading===========")
e_trade_gym = AssetTradingEnv(df=test_df, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ppo, environment = e_trade_gym
)
# df_account_value.to_csv(
#     "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
# )
# df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

print("==============Get Backtest Results===========")
# perf_stats_all = backtest_stats(df_account_value)
# perf_stats_all = pd.DataFrame(perf_stats_all)
# perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

returns_test_df = get_returns_df(test_df)
strategy_returns = calc_strategy_returns(df_actions, returns_test_df, save_csv=False)
strategy_metrics = calc_metrics_by_year(strategy_returns, df_actions, save_csv=False)
print(strategy_metrics)

print('Runtime: ' + str((time.time() - start_time)/60))
