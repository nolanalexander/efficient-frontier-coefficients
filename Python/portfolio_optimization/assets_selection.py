import pandas as pd
import numpy as np

from data_processing.read_in_data import read_in_assets_data
from data_processing.data_cleaning import multiindex_df_along_one_col
from ts_features.feature_selection import multicollinearity_tables, pca_plots

'''
Suite of common asset selection methods (PCA, correlation table)
'''

def assets_selection_suite(tickers, start_date, end_date):
    assets_df = read_in_assets_data('', tickers, start_date, end_date, save_csv=False)
    assets_returns_df = multiindex_df_along_one_col(assets_df, 'Chg')
    corr_mat, corr_df = multicollinearity_tables(assets_returns_df)
    print(corr_df.head(10))
    pca_plots(assets_returns_df, save_dir=None)
    return corr_mat, corr_df


