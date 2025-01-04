import pandas as pd
import numpy as np
from data_processing.data_cleaning import add_all_multiindex_combinations

'''
Preprocess by cleaning/tranforming and splitting data
'''

# Cleans/transforms data and adds predictand     
def preprocess(assets_df_wmiss, test_start_date, predictand_name, days_fwd=1, ema_alpha=None, is_rl=False, use_ad_osc=False):
    
    assets_df_wmiss = assets_df_wmiss.sort_values(['Ticker', 'Date'])
    
    # Adds ema smoothing
    if(ema_alpha is not None):
        assets_df_wmiss[~assets_df_wmiss.columns.isin(['Date', 'Ticker'])] = assets_df_wmiss[~assets_df_wmiss.columns.isin(['Date', 'Ticker'])].ewm(alpha=ema_alpha).mean()
    
    fwd_chgs = assets_df_wmiss.groupby('Ticker')['Adj Close'].pct_change(days_fwd).shift(-days_fwd).values
    assets_df_wmiss = assets_df_wmiss.drop(columns=['Adj Close'])
    if(predictand_name == 'Next_Return_Binary'):
        assets_df_wmiss[predictand_name] =  ["Up" if fwd_chg > 0 else "Down" for fwd_chg in fwd_chgs]
    elif(predictand_name == 'Next_Return'):
        assets_df_wmiss[predictand_name] =  fwd_chgs
    elif(predictand_name is None):
        pass
    else:
        raise ValueError(predictand_name + ' is not a valid dependent variable. Select from \'Return_Binary\', \'Return\', or None ')
    
    # Handle days that had the same high and low
    if use_ad_osc:
        tickers_with_inf = assets_df_wmiss['Ticker'][np.isinf(assets_df_wmiss['A/D_Osci'])].unique()
        assets_df_wmiss = assets_df_wmiss[~assets_df_wmiss['Ticker'].isin(tickers_with_inf)]
    else:
        assets_df_wmiss = assets_df_wmiss.drop(columns=['A/D_Osci', 'A/D_Osci_10D_MA'])
    
    # Handles missing values and edgecases when the last date has bad data
    assets_df = assets_df_wmiss.dropna()
    row_is_last_date = (assets_df['Date'] == assets_df['Date'].max())
    if(sum(row_is_last_date) != len(assets_df['Date'].unique())):
        assets_df = assets_df[~row_is_last_date]
    assets_df = assets_df.reset_index(drop=True)
    assets_df['Date'] = pd.to_datetime(assets_df['Date'])
    assets_df = assets_df.set_index(['Date', 'Ticker'])
    
    if is_rl:
        processed_full_df = add_all_multiindex_combinations(assets_df, fillna_val=0)
    else:
        processed_full_df = assets_df
        
    # Only use Fridays if weekly to remove overlap
    if(days_fwd == 5):
        processed_full_df = processed_full_df[processed_full_df.index.get_level_values('Date').dt.weekday == 4]
    return processed_full_df

def train_test_split(df, predictand, predictors, test_start_date):
    train_df = df[df.index.get_level_values(level='Date') < test_start_date]
    test_df = df[df.index.get_level_values(level='Date') >= test_start_date]
    X_train, y_train = train_df[predictors], train_df[predictand]
    X_test, y_test = test_df[predictors], test_df[predictand]
    return X_train, y_train, X_test, y_test

def ts_train_test_split(df : pd.DataFrame, test_start_date, date_name='Date'):
    if df.index.nlevels > 1:
        date_index = df.index.get_level_values(date_name)
    else:
        date_index = df.index
    train_df, test_df = df[date_index< test_start_date].copy(), df[date_index >= test_start_date].copy()
    return train_df, test_df

def preprocess_and_save_data(assets_set, test_start_date, predictand_name, days_fwd=1, ema_alpha=None, is_rl=False):
    proc_data_dir = '../Portfolios/'+assets_set+'/Assets_Data/Processed_Data/'
    pred_df = pd.read_csv(proc_data_dir + 'assets_data_ts_features.csv', parse_dates=['Date'])
    processed_df = preprocess(pred_df, test_start_date, predictand_name, days_fwd=days_fwd, ema_alpha=ema_alpha, is_rl=is_rl)
    processed_df.to_csv(proc_data_dir + predictand_name.lower() + '_processed_assets_data_ts_features.csv', index=True)
    
    