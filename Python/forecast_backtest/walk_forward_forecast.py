import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, r2_score
from tqdm import tqdm
from forecast_backtest.ForecastModel import ForecastModel

from data_processing.preprocess_data import train_test_split
from forecast_backtest.forecast_metrics import rmse, mae, mad, aicc, bic

'''
Implements a walk-forward forcaster with variable train/tune updates.
The model can be applied to data for all dates and tickers or can be separated by ticker.
'''

def get_forecast_metrics(forecast_df, y_test, is_continuous, separate_tickers=False):
    if separate_tickers:
        forecast_df_and_y_test = forecast_df.copy()
        forecast_df_and_y_test['y_test'] = y_test
        forecast_metrics = forecast_df_and_y_test.groupby('Ticker').apply(get_indiv_forecast_metrics, is_continuous)
        forecast_metrics.loc['Full'] = get_full_forecast_metrics(forecast_df, y_test, is_continuous)
    else:
        forecast_metrics = get_full_forecast_metrics(forecast_df, y_test, is_continuous)
    return forecast_metrics

def get_indiv_forecast_metrics(forecast_df_and_y_test, is_continuous):
    return get_full_forecast_metrics(forecast_df_and_y_test.drop(columns=['y_test']), 
                                      forecast_df_and_y_test['y_test'], is_continuous)

def get_full_forecast_metrics(forecast_df, y_test, is_continuous):
    if is_continuous:
        forecast_metrics = pd.Series({'R2'      : r2_score(y_test.values, forecast_df['Forecast'].values), 
                                      'RMSE'    : rmse(y_test.values, forecast_df['Forecast'].values), 
                                      'MAE'     : mae(y_test.values, forecast_df['Forecast'].values),
                                      'MAD'     : mad(y_test.values, forecast_df['Forecast'].values),
                                      'Avg_AICc' : forecast_df['AICc'].mean(),
                                      'Avg_BIC' : forecast_df['BIC'].mean() })
    else:
        forecast_metrics = pd.Series({'Accuracy'  : np.mean((y_test.values == forecast_df['Discrete_Forecast'].values)), 
                                      'Precision' : precision_score(y_test.values, forecast_df['Discrete_Forecast'].values, pos_label='Up'), 
                                      'Recall'    : recall_score(y_test.values, forecast_df['Discrete_Forecast'].values, pos_label='Up'),
                                      'F1'        : f1_score(y_test.values, forecast_df['Discrete_Forecast'].values, pos_label='Up'),
                                      'AUC'       : roc_auc_score(y_test.values, forecast_df['Forecast'].values), })
    return forecast_metrics

def get_in_samp_metrics(is_continuous, y_train, y_pred, k, n, y_pred_disc=None):
    if is_continuous:
        is_metrics = [r2_score(y_train.values, y_pred),
                      rmse(y_train.values, y_pred),
                      mae(y_train.values, y_pred),
                      mad(y_train.values, y_pred),
                      aicc(y_train.values, y_pred, k, n),
                      bic(y_train.values, y_pred, k, n)]
    else:
        if y_pred_disc is None:
            raise ValueError('y is not continuous and discrete y_pred is not supplied')
        is_metrics = [np.mean(y_train.values == y_pred_disc),
                      precision_score(y_train.values, y_pred_disc, pos_label='Up'),
                      recall_score(y_train.values, y_pred_disc, pos_label='Up'),
                      roc_auc_score(y_train.values, y_pred),
                      f1_score(y_train.values, y_pred_disc, pos_label='Up')]
    return is_metrics

# Handles appending to a multi-index df
def append_date_to_dateticker_data(data, test_date, new_data, is_df=True):
    data = data.copy()
    for ix in new_data.index:
        if is_df:
            data.loc[ix, :] = new_data.loc[ix, :].copy()
        else: # is series
            data.loc[ix] = new_data.loc[ix]
    return data.sort_index(level=['Date', 'Ticker'])

# Converts update frequency (or proportion) to the update dates
def get_update_test_dates(train_dates, test_dates, update_freq_prop=None, update_freq=None):        
    if (update_freq_prop is None or np.isinf(update_freq_prop)) and update_freq is None:
        update_dates = test_dates[[0]]
    else:
        if update_freq is None:
            update_freq_prop = max(update_freq_prop, 1 / len(train_dates))
            update_freq = round(len(train_dates) * update_freq_prop)
        sorted_test_dates = pd.Series(test_dates.copy()).sort_values(ascending=True)
        ix = np.arange(0, len(sorted_test_dates))
        sorted_test_dates.index = ix
        update_dates = sorted_test_dates[ix[ix % update_freq == 0]]
    return update_dates

'''
The Walk-Forward Forecaster
separate_tickers : Whether to run a separate model for each ticker
retrain_freq_proportion : The proportion of the train data needed to be observed from the test set to retrain
retune_freq_proportion : The proportion of the train data needed to be observed from the test set to retune
retrain_freq : The number of observations needed to be observed from the test set to retrain
retune_freq : The number of observations needed to be observed from the test set to retune
data_is_dateticker : Whether the data is a multiindex of Date x Ticker, otherwise is indexed by Date for one Ticker
'''

def walk_forward_forecast(df, predictand_name, predictor_names, test_start_date, forecast_model : ForecastModel,
                          separate_tickers=False, retrain_freq_proportion=0.025, retune_freq_proportion=0.5, retrain_freq=None, retune_freq=None, 
                          n_splits=10, scoring='roc_auc', param_grid=None, rolling_window=None, data_is_dateticker=True):
    if separate_tickers and not data_is_dateticker:
        raise ValueError('separating_ticker = True but data_is_dateticker = False')
    elif separate_tickers and data_is_dateticker:
        forecast_df = df.groupby(level='Ticker').apply(indiv_walk_forward_forecast, predictand_name, predictor_names, test_start_date, forecast_model, 
                                                       retrain_freq_proportion=retrain_freq_proportion, retune_freq_proportion=retune_freq_proportion, 
                                                       retrain_freq=retrain_freq, retune_freq=retune_freq,
                                                       n_splits=n_splits, scoring=scoring, param_grid=param_grid, rolling_window=rolling_window, data_is_dateticker=data_is_dateticker, 
                                                       unique_tickers=df.index.get_level_values('Ticker').unique())
        forecast_df = forecast_df.droplevel(0)
    else:
        forecast_df = indiv_walk_forward_forecast(df, predictand_name, predictor_names, test_start_date, forecast_model, 
                                                  retrain_freq_proportion=retrain_freq_proportion, retune_freq_proportion=retune_freq_proportion, 
                                                  retrain_freq=retrain_freq, retune_freq=retune_freq,
                                                  n_splits=n_splits, scoring=scoring, param_grid=param_grid, rolling_window=rolling_window, data_is_dateticker=data_is_dateticker)
    return forecast_df

def indiv_walk_forward_forecast(df, predictand_name, predictor_names, test_start_date, forecast_model : ForecastModel,
                                retrain_freq_proportion=0.025, retune_freq_proportion=0.5, retrain_freq=None, retune_freq=None, 
                                n_splits=10, scoring='roc_auc', param_grid=None, rolling_window=None, data_is_dateticker=True, unique_tickers=None):
    X_train, y_train, X_test, y_test = train_test_split(df, predictand_name, predictor_names, test_start_date)
    is_continuous = (y_train.dtype.name != 'category')
    if is_continuous != forecast_model.is_continuous:
        raise ValueError(f'Data is_continuous = {is_continuous}, but forecast_model.is_continuous = {forecast_model.is_continuous}')
    metric_names = ['R2', 'RMSE', 'MAE', 'MAD', 'AICc', 'BIC'] if is_continuous else ['Discrete_Forecast', 'Accuracy', 'Precision', 'Recall', 'AUC', 'F1']
    forecast_df = pd.DataFrame(columns=['Forecast'] + metric_names, index=X_test.index)
    
    # Set up retrain and retune dates
    train_dates = X_train.index.get_level_values('Date').unique()
    test_dates = X_test.index.get_level_values('Date').unique()
    retrain_dates = get_update_test_dates(train_dates, test_dates, retrain_freq_proportion, retrain_freq)
    retune_dates = get_update_test_dates(train_dates, test_dates, retune_freq_proportion, retune_freq)
    retrain_dates = retrain_dates.combine_first(retune_dates)
    
    if unique_tickers is not None:
        cur_ticker = df.index.get_level_values('Ticker')[0]
        ticker_ix = np.where(unique_tickers == cur_ticker)[0][0] + 1
        
    # Rolling/Expanding forecast
    p_bar = tqdm(test_dates.values)
    for test_date in p_bar:
        if unique_tickers is not None:
            p_bar.set_description(f'Ticker {ticker_ix}/{len(unique_tickers)} {cur_ticker} on {pd.to_datetime(test_date).date()} ')
        else:
            p_bar.set_description(f'{pd.to_datetime(test_date).date()} ')
        
        # Update Window
        if rolling_window is not None:
            if data_is_dateticker:
                train_dates = X_train.index.get_level_values('Date').unique()
                earliest_date = train_dates[-rolling_window:][0]
                X_train = X_train.loc[X_train.index.get_level_values('Date') >= earliest_date]
                y_train = y_train.loc[y_train.index.get_level_values('Date') >= earliest_date]
            else:
                X_train, y_train = X_train.iloc[-rolling_window:], y_train.iloc[-rolling_window:]
        # Tune
        if forecast_model.has_hyperparams and (param_grid is not None) and (test_date in retune_dates.values):
            forecast_model, best_params = forecast_model.tune(X_train, y_train, param_grid, n_splits, scoring)
        # Fit
        if test_date in retrain_dates.values:
            forecast_model.fit(X_train, y_train)
        # Forecast
        cur_X_test = X_test.loc[[test_date]]
        model_forecast = forecast_model.forecast(cur_X_test)
        model_forecast_disc = forecast_model.forecast(cur_X_test, return_proba=False)
        y_pred = forecast_model.forecast(X_train)
        y_pred_disc = forecast_model.forecast(X_train, return_proba=False) if not is_continuous else None
        k, n = len(X_train.columns), len(X_train.index)+1
        if data_is_dateticker:
            model_forecast = pd.Series(model_forecast, index=cur_X_test.index.get_level_values('Ticker'))
            model_forecast_disc = pd.Series(model_forecast_disc, index=cur_X_test.index.get_level_values('Ticker'))
            in_samp_metrics = get_in_samp_metrics(is_continuous, y_train, y_pred, k, n, y_pred_disc=y_pred_disc)
            for ticker in cur_X_test.index.get_level_values('Ticker'):
                forecast_df.loc[(test_date, ticker)] = ([model_forecast[ticker]] + ([model_forecast_disc[ticker]] if not is_continuous else []) + 
                                                        in_samp_metrics)
        else:
            forecast_df.loc[test_date] = ([model_forecast[0]] + ([model_forecast_disc[0]] if not is_continuous else []) + 
                                          get_in_samp_metrics(is_continuous, y_train, y_pred, k, n, y_pred_disc=y_pred_disc))
        # Walk forward
        if data_is_dateticker:
            X_train = append_date_to_dateticker_data(X_train, test_date, X_test.loc[[test_date]], is_df=True)
            y_train = append_date_to_dateticker_data(y_train, test_date, y_test.loc[[test_date]], is_df=False)
        else:
            X_train.loc[test_date] = X_test.loc[test_date]
            y_train.loc[test_date] = y_train.loc[test_date]
    return forecast_df

def select_features_from_mrmr_weights(mrmr_weights, min_feature_weight=None, max_features=None):
    if min_feature_weight is not None:
        predictor_names = mrmr_weights.index[mrmr_weights >= min_feature_weight]
        print(f'Keeping {len(predictor_names)} / {len(mrmr_weights)} features from selecting MRMR weights >= {min_feature_weight}.')
    elif max_features is not None:
        predictor_names = mrmr_weights.index[:max_features]
        print(f'Keeping {len(predictor_names)} / {len(mrmr_weights)} features from selecting top {max_features} MRMR weighted features.')
    else:
        min_feature_weight = 1 / len(mrmr_weights)
        predictor_names = mrmr_weights.index[mrmr_weights >= min_feature_weight]
        print(f'Keeping {len(predictor_names)} / {len(mrmr_weights)} features from selecting MRMR weights >= equal weights.')
    return predictor_names

def run_walk_forward_forecast(assets_set, predictand_name, test_start_date, forecast_model, version_name, is_continuous, 
                               min_feature_weight=None, max_features=None, separate_tickers=False,
                               retrain_freq_proportion=0.025, retune_freq_proportion=1, retrain_freq=None, retune_freq=None, 
                               n_splits=10, scoring='roc_auc', param_grid=None, rolling_window=None, data_is_dateticker=True):
    start_time = time.time()
    proc_data_dir = '../Portfolios/'+assets_set+'/Assets_Data/Processed_Data/'
    processed_df = pd.read_csv(proc_data_dir + predictand_name.lower() + '_processed_assets_data_ts_features.csv', index_col=['Date', 'Ticker'], parse_dates=True)
    processed_df = processed_df.sort_index(level=['Date', 'Ticker'])
    mrmr_weights = pd.read_csv('../Portfolios/'+assets_set+'/Feature_Selection/features_MRMR_weights.csv', index_col=0)['MRMR_Weights']
    predictor_names = select_features_from_mrmr_weights(mrmr_weights, min_feature_weight=min_feature_weight, max_features=max_features)
        
    X_train, y_train, X_test, y_test = train_test_split(processed_df, predictand_name, predictor_names, test_start_date)
    if not is_continuous:
        y_train, y_test = y_train.astype('category'), y_test.astype('category')
    forecast_df = walk_forward_forecast(processed_df, predictand_name, predictor_names, test_start_date, forecast_model,
                                        separate_tickers=separate_tickers,
                                        retrain_freq_proportion=retrain_freq_proportion, retune_freq_proportion=retune_freq_proportion, 
                                        retrain_freq=retrain_freq, retune_freq=retune_freq,
                                        n_splits=n_splits, scoring=scoring, param_grid=param_grid, rolling_window=rolling_window)
    forecast_metrics = get_forecast_metrics(forecast_df, y_test, is_continuous, separate_tickers=data_is_dateticker)
    forecast_backtest_dir = '../Portfolios/'+assets_set+'/Forecast_Backtest/' + predictand_name + '/' + version_name + '/'
    if not os.path.exists(forecast_backtest_dir):
        os.makedirs(forecast_backtest_dir)
    forecast_df.to_csv(forecast_backtest_dir + predictand_name.lower() + '_forecast_df.csv', index=True)
    forecast_metrics.to_csv(forecast_backtest_dir + predictand_name.lower() + '_forecast_metrics.csv', index=True)
    
    print('Walk-forward Runtime:', round((time.time() - start_time)/60), 'mins' )

