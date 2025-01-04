import pandas as pd
import numpy as np

from data_processing.preprocess_data import preprocess_and_save_data
from ts_features.feature_engr import create_pred_data
from ts_features.feature_selection import feature_selection_suite
from forecast_backtest.ForecastModel import RFForecastModel, LassoForecastModel
from forecast_backtest.walk_forward_forecast import run_walk_forward_forecast
from forecast_backtest.backtest_forecasts import run_backtest_forecasts


run_create_preprocessed_data  = 0
run_feature_selection_suite   = 0
run_run_walk_forward_forecast = 1
run_run_backtest_forecasts    = 1

assets_sets = ['Growth_Val_Mkt_Cap', 'SPDR_ETF_Sectors']
# assets_sets = ['SPDR_ETF_Sectors']
# assets_sets = ['Growth_Val_Mkt_Cap']
assets_sets_settings_df = pd.read_csv('asset_sets_settings.csv', index_col=0, parse_dates=['Start_Date', 'Test_Start_Date', 'End_Date'])
assets_sets_settings_df['Assets'] = [assets_set.split(' ') for assets_set in assets_sets_settings_df['Assets']]

predictand_name, is_continuous = 'Next_Return_Binary', False
version_name = 'RF'
forecast_model = RFForecastModel()
param_grid = {'n_estimators'     : [100],
              # 'max_features'     : np.arange(0.2, 1, 0.1),
              'max_depth'        : [3, 5, 7, 9] }
              # 'min_samples_leaf' : np.arange(1, 10, 2),
              # 'max_samples'      : [0.3, 0.5, 0.8] }
scoring='roc_auc'
              
predictand_name, is_continuous = 'Next_Return', True
version_name = 'LASSO'
forecast_model = LassoForecastModel()
param_grid = {'alpha' : [0, 1e-15, 1e-10, 1e-5, 1e-1, 1]}
scoring='neg_root_mean_squared_error'

for assets_set in assets_sets:
    assets, start_date, test_start_date, end_date = assets_sets_settings_df.loc[assets_set]
    if run_create_preprocessed_data:
        create_pred_data(assets_set, include_yearly=False)
        preprocess_and_save_data(assets_set, test_start_date, predictand_name, days_fwd=1)
    if run_feature_selection_suite:
        feature_selection_suite(assets_set, predictand_name, test_start_date, max_corr=0.8, is_continuous=is_continuous)
    if run_run_walk_forward_forecast:
        run_walk_forward_forecast(assets_set, predictand_name, test_start_date, forecast_model, version_name, is_continuous, 
                                  min_feature_weight=None, separate_tickers=False,
                                  retrain_freq_proportion=0.025, retune_freq_proportion=0.5, retrain_freq=None, retune_freq=None,
                                  n_splits=5, scoring=scoring, param_grid=param_grid, rolling_window=None)
    if run_run_backtest_forecasts:
        run_backtest_forecasts(assets_set, predictand_name, forecast_model, version_name, is_continuous,
                               long_short=False, max_leverage=1.5, discrete_prediction=False, transaction_cost=0.01)
        
        