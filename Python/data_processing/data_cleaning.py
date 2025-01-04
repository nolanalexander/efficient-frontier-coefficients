import pandas as pd
import numpy as np
from itertools import product

'''
Data cleaning utility functions including filling in missing data, 
and handling multi-index dfs
'''

def fill_nan_ts_df(ts_df, method='nearest'):
    return ts_df.apply(fill_nan_ts, method=method)

def fill_nan_ts(ts, method='nearest'):
    if(method == 'nearest'):
        ts[ts.isna()] = ts.iloc[ts[ts.isna()].index.get_indexer(pd.to_datetime(ts[ts.isna()].index), method='nearest')].values
    elif(method == 'zero'):
        ts[ts.isna()] = 0
    else:
        raise ValueError(method + ' is not a valid data fill method')
    return ts

def fill_miss_ts_df(ts_df, all_dates, method='nearest', keep_dates_not_in_all_dates=False):
    ts_df_clean = ts_df.copy()
    all_dates = pd.to_datetime(all_dates)
    missing_dates = all_dates[~np.isin(all_dates, pd.to_datetime(ts_df.index.get_level_values('Date')))]
    if(method == 'nearest'):
        ts_df_nearest_miss = ts_df_clean.iloc[ts_df_clean.index.get_indexer(missing_dates, method='nearest')]
        ts_df_nearest_miss.index = missing_dates
        ts_df_clean = pd.concat([ts_df_clean, ts_df_nearest_miss])
    elif(method == 'zero'):
        for missing_date in missing_dates:
            ts_df_clean.loc[missing_date] = 0
    else:
        raise ValueError(method + ' is not a valid data fill method')
    
    if not keep_dates_not_in_all_dates:
        ts_df_clean = ts_df_clean[ts_df_clean.index.isin(all_dates)]
    return ts_df_clean.sort_index()

'''
Converts a multi-index df to a single-index df subset to one column
with rows being the level 0 index and cols being the level 1 index
'''
def multiindex_df_along_one_col(multiindex_df, column, col_level=0):
    df = multiindex_df[[column]].unstack(level=col_level)[1:].copy()
    df.columns = df.columns.levels[1].values
    return df

def add_all_multiindex_combinations(df, fillna_val=0):
    index_lv0_name, index_lv1_name = df.index.names
    combination = list(product(df.index.get_level_values(index_lv0_name).unique(), 
                               df.index.get_level_values(index_lv1_name).unique()))
    comb_df = pd.DataFrame(combination, columns=[index_lv0_name, index_lv1_name])
    all_combs_df = comb_df.merge(df.reset_index(), on=[index_lv0_name, index_lv1_name], how='left')
    all_combs_df = all_combs_df.sort_values([index_lv0_name, index_lv1_name]).fillna(fillna_val)
    all_combs_df = all_combs_df.set_index([index_lv0_name, index_lv1_name])
    return all_combs_df

def ts_train_test_split(df : pd.DataFrame, test_start_date, date_name='Date'):
    if df.index.nlevels > 1:
        date_index = df.index.get_level_values(date_name)
    else:
        date_index = df.index
    train_df, test_df = df[date_index< test_start_date].copy(), df[date_index >= test_start_date].copy()
    return train_df, test_df
