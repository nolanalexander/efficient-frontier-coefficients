import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import datetime as dt
import time

from data_processing.data_cleaning import fill_miss_ts_df
from data_processing.ts_utils import get_quarters, get_month_first_dates
from data_processing.read_in_data import read_in_eq_weight, read_in_fama_french


def create_mkt_forecast_data(assets_set, start_date, end_date):
    start_time = time.time()
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    assets_returns_df = assets_returns_df[(assets_returns_df.index >= start_date) & (assets_returns_df.index <= end_date)]
    eq_weight_df = read_in_eq_weight(assets_set, assets_returns_df.index)
    
    # Predictand DF
    eq_weight_chgs = eq_weight_df['Chg']
    eq_weight_chgs_yearly = eq_weight_chgs.copy()
    eq_weight_chgs_quarterly = eq_weight_chgs.copy()
    eq_weight_chgs_monthly = eq_weight_chgs.copy()
    
    eq_weight_chgs_yearly.index = eq_weight_chgs_yearly.index.year
    eq_weight_chgs_quarterly.index = get_quarters(eq_weight_chgs_quarterly.index)
    eq_weight_chgs_monthly.index = get_month_first_dates(eq_weight_chgs_monthly.index)
    
    def get_predictand_df(mkt_chg_df):
        mkt_chgs_interval = mkt_chg_df.copy().groupby(level=0).mean()
        predictand_df = pd.DataFrame({'Mkt_Next_Up' : [1 if above0 else 0 for above0 in (mkt_chgs_interval > 0).shift(-1).iloc[:-1]],
                                      'Mkt_Next_Down' : [1 if below0 else 0 for below0 in (mkt_chgs_interval <= 0).shift(-1).iloc[:-1]],
                                      'Mkt_Next_Abs_Return' : mkt_chgs_interval.shift(-1).abs()
                                      }, index=mkt_chgs_interval.index[:-1])
        return predictand_df
    
    predictand_df_yearly = get_predictand_df(eq_weight_chgs_yearly)
    predictand_df_quarterly = get_predictand_df(eq_weight_chgs_quarterly)
    predictand_df_monthly = get_predictand_df(eq_weight_chgs_monthly)
    predictand_df_rolling = get_predictand_df(eq_weight_chgs)
    
    ef_coefs_df_yearly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_yearly.csv', index_col=0)[['r_MVP', 'sigma_MVP', 'u']]
    ef_coefs_df_quarterly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_quarterly.csv', index_col=0, parse_dates=True)[['r_MVP', 'sigma_MVP', 'u']]
    ef_coefs_df_monthly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_monthly.csv', index_col=0, parse_dates=True)[['r_MVP', 'sigma_MVP', 'u']]
    ef_coefs_df_rolling_1mo = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_rolling_1mo.csv', index_col=0, parse_dates=True)[['r_MVP', 'sigma_MVP', 'u']]
    ef_coefs_df_rolling_3mo = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_rolling_3mo.csv', index_col=0, parse_dates=True)[['r_MVP', 'sigma_MVP', 'u']]
    
    def add_lags(df, lags=1):
        for lag in range(1,lags+1):
            for col in df.columns:
                df[col+'_'+str(lag)+'lag'] = df[col].shift(lag)
        return df
    ef_coefs_df_yearly = add_lags(ef_coefs_df_yearly)
    ef_coefs_df_quarterly = add_lags(ef_coefs_df_quarterly)
    ef_coefs_df_monthly = add_lags(ef_coefs_df_monthly)
    ef_coefs_df_rolling_1mo = add_lags(ef_coefs_df_rolling_1mo)
    ef_coefs_df_rolling_3mo = add_lags(ef_coefs_df_rolling_3mo)
    
    def add_tech_indicators(asset_df):
        asset_df['Lowest_Low_10D'] = asset_df['Low'].rolling(int(10)).min().values
        asset_df['Highest_High_10D'] = asset_df['High'].rolling(int(10)).max().values
        asset_df['Sto_K'] = 100*(asset_df['Close'] - asset_df['Lowest_Low_10D'])/(asset_df['Highest_High_10D'] - asset_df['Lowest_Low_10D'])
        asset_df['Williams_R'] = (100*(asset_df['Highest_High_10D'] - asset_df['Close'])
                                       /(asset_df['Highest_High_10D'] - asset_df['Lowest_Low_10D']))
        asset_df.drop(columns=['Lowest_Low_10D', 'Highest_High_10D'], inplace=True)
        
        asset_df['Adj_Close_10D_lag'] = asset_df['Adj Close'].shift(10).values
        asset_df['Adj_Close_5D_MA'] = asset_df['Adj Close'].rolling(5).mean().values
        asset_df['Adj_Close_10D_MA'] = asset_df['Adj Close'].rolling(10).mean().values
        
        asset_df['Momentum'] = (asset_df['Adj Close'] - asset_df['Adj_Close_10D_lag'])
        def pos_mean(x): return x[x>0].mean()
        def neg_mean(x): return -x[x<0].mean()
        asset_df['Chg'] = asset_df['Adj Close'].pct_change()
        asset_df['RSI'] = 100 - (100/(1+asset_df['Chg'].rolling(10).apply(pos_mean)/asset_df['Chg'].rolling(10).apply(neg_mean)))
        # asset_df['ROC'] = 100*(asset_df['Adj Close'] / asset_df['Adj_Close_10D_lag'])
        # asset_df['A/D_Osci'] = (asset_df['High']-asset_df['Close'].shift(1))/(asset_df['High'] - asset_df['Low'])
        asset_df['Disparity_5D'] = 100*(asset_df['Adj Close'] / asset_df['Adj_Close_5D_MA'])
        asset_df['Disparity_10D'] = 100*(asset_df['Adj Close'] / asset_df['Adj_Close_10D_MA'])
        # asset_df['OSCP'] = ((asset_df['Disparity_5D'] - asset_df['Disparity_10D'])/asset_df['Disparity_5D'])
        asset_df['M'] = ((asset_df['High']+asset_df['Low']+asset_df['Close'])/3.0)
        asset_df['SM'] = asset_df['M'].rolling(10).mean().values
        asset_df['D'] = asset_df['SM'].rolling(10).std().values
        asset_df['CCI'] = (asset_df['M']-asset_df['SM'])/(asset_df['D']*.015)
        asset_df.drop(columns=['Disparity_5D', 'Disparity_10D', 'Adj_Close_5D_MA', 'Adj_Close_10D_lag',
                                'Adj_Close_10D_MA', 'M', 'SM', 'D', 
                                'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Chg'], inplace=True)
        asset_df.iloc[10:] = asset_df.iloc[10:].fillna(50)
        return asset_df
    
    # Create technical indicators DF
    yf.pdr_override()
    spx_df_for_tech_ind = pdr.get_data_yahoo('^SP500TR', start_date - dt.timedelta(30), end_date)
    spx_df_for_tech_ind = fill_miss_ts_df(spx_df_for_tech_ind, assets_returns_df.index, keep_dates_not_in_all_dates=True)
    tech_ind_df = add_tech_indicators(spx_df_for_tech_ind.copy())
    tech_ind_df = tech_ind_df[tech_ind_df.index >= start_date]
    # print(tech_ind_df.corr())
    
    # Read in Fama French Data
    # ff_df = pd.read_csv('../CSV_Data/F-F_Research_Data_Factors_daily.csv', 
    #                         skiprows=4, skipfooter=2, engine='python', index_col=0)[['Mkt-RF', 'SMB', 'HML']]
    # ff_df.index = pd.to_datetime(ff_df.index, format='%Y%m%d')
    # ff_df = ff_df[(ff_df.index >= start_date) & (ff_df.index <= end_date)]
    # ff_df.index = ff_df.index.rename('Date')
    ff_df = read_in_fama_french(assets_returns_df.index)
    
    predictor_df = tech_ind_df.merge(ff_df, left_index=True, right_index=True)
    predictor_df['Year'] = predictor_df.index.year
    predictor_df['Quarter'] = get_quarters(predictor_df.index)
    predictor_df['Month'] = [dt.datetime(date.year, date.month, 1) for date in predictor_df.index]
    predictor_df_yearly = predictor_df.drop(['Quarter', 'Month'], axis=1).groupby('Year').mean()
    predictor_df_yearly = predictor_df_yearly.merge(ef_coefs_df_yearly, left_index=True, right_index=True)
    predictor_df_quarterly = predictor_df.drop(['Year', 'Month'], axis=1).groupby('Quarter').mean()
    predictor_df_quarterly = predictor_df_quarterly.merge(ef_coefs_df_quarterly, left_index=True, right_index=True)
    predictor_df_monthly = predictor_df.drop(['Year', 'Quarter'], axis=1).groupby('Month').mean()
    predictor_df_monthly = predictor_df_monthly.merge(ef_coefs_df_monthly, left_index=True, right_index=True)
    predictor_df_rolling_1mo = predictor_df.drop(['Year', 'Quarter', 'Month'], axis=1)
    predictor_df_rolling_1mo = predictor_df_rolling_1mo.merge(ef_coefs_df_rolling_1mo, left_index=True, right_index=True)
    predictor_df_rolling_3mo = predictor_df.drop(['Year', 'Quarter', 'Month'], axis=1)
    predictor_df_rolling_3mo = predictor_df_rolling_3mo.merge(ef_coefs_df_rolling_3mo, left_index=True, right_index=True)
    
    full_df_yearly = predictand_df_yearly.merge(predictor_df_yearly, right_index=True, left_index=True, how='outer')
    full_df_yearly.to_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_yearly.csv')
    full_df_quarterly = predictand_df_quarterly.merge(predictor_df_quarterly, right_index=True, left_index=True, how='outer')
    full_df_quarterly.to_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_quarterly.csv')
    full_df_monthly = predictand_df_monthly.merge(predictor_df_monthly, right_index=True, left_index=True, how='outer')
    full_df_monthly.to_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_monthly.csv')
    full_df_rolling_1mo = predictand_df_rolling.merge(predictor_df_rolling_1mo, right_index=True, left_index=True, how='outer')
    full_df_rolling_1mo.to_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_rolling_1mo.csv')
    full_df_rolling_3mo = predictand_df_rolling.merge(predictor_df_rolling_1mo, right_index=True, left_index=True, how='outer')
    full_df_rolling_3mo.to_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_rolling_1mo.csv')
    print('Create Market Forecast Data Runtime:', round((time.time() - start_time)/60), 'mins' )
    return full_df_monthly






