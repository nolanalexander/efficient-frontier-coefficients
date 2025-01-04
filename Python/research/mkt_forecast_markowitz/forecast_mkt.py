import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from graphviz import Source
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from tqdm import tqdm

from forecast_backtest.forecast_metrics import npv
from data_processing.ts_utils import get_prev_months_first_date, get_next_month_first_date

'''
Provides walk-forward forecasts of whether the market will be 
up or down each day using a classifier model.

Note: the date saved is the date the forecast is run on,
not the date being forecasted.
'''

def rolling_binary_forecast(assets_set, model_name, predictand, predictors, df, predictors_name, time_interval, 
                            test_start_date, retrain_freq=1, retune_freq=1, is_rolling=True, n_splits=10, param_grid=None, save_trees=False,
                            scoring='roc_auc', verbose=False,
                            expanding=False, use_sample_weight=False, use_ts_split=False):
    test_start_date = test_start_date.year if time_interval == 'Yearly' else test_start_date
    prev_test_start_date = get_prev_months_first_date(test_start_date)[0] if not is_rolling else test_start_date
    sample_weight = df['Mkt_Next_Abs_Return']
    train_df = df.loc[df.index < prev_test_start_date, predictors + [predictand]].dropna().copy()
    test_df = df.loc[df.index >= prev_test_start_date, predictors + [predictand]].copy()
    sample_weight_train = sample_weight[sample_weight.index < prev_test_start_date].loc[train_df.index]
    sample_weight_test = sample_weight[sample_weight.index >= prev_test_start_date]
    
    X_train, y_train = train_df[predictors].copy(), train_df[predictand].copy()
    X_test, y_test = test_df[predictors].copy(), test_df[predictand].copy()
    
    non_tuning_models = ['Logistic']
    model_by_name = {'CART'        : DecisionTreeClassifier(random_state=0),
                     'RF'          : RandomForestClassifier(n_estimators=100, random_state=0),
                     'Logistic'    : LogisticRegression(penalty='none', random_state=0),
                     'L2_Logistic' : LogisticRegression(penalty='l2', random_state=0),
                     'EN_Logistic' : LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1e4, random_state=0) }
    model = model_by_name[model_name]
    
    forecast_df = pd.DataFrame(columns=['Forecast', 'Prob', 'Accuracy', 'Precision', 'Recall', 'NPV', 'AUC', 'F1'])
    forecast_dir = '../Portfolios/'+assets_set+'/Mkt_Forecast/'
    
    ## Rolling forecast
    # ix = np.arange(0,len(test_df.index))
    # retune_freq = round(len(train_df.index) / len(train_df.index) )
    # retune_dates = test_df.index[ix[ix % retune_freq == 0]]
    # retune_dates = pd.Series(test_df.index).groupby(test_df.index.year).first()
    # retune_dates = test_df.index[[0]]
    
    retrain_counter, retune_counter = 0, 0
    p_bar = tqdm(test_df.index)
    coefs_df = pd.DataFrame(columns=list(predictors) + ['intercept'], index=test_df.index)
    for test_date in p_bar:
        p_bar.set_description(f'{pd.to_datetime(test_date).date()} ')
        
        cur_X_test = X_test.loc[[test_date]]
        if (retrain_counter >= retrain_freq or test_date == test_df.index[0]):
            retrain_counter = 0
            
            if model_name not in non_tuning_models and (retune_counter >= retune_freq or test_date == test_df.index[0]):
                retune_counter = 0
                if use_ts_split:
                    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=int(len(X_train.index)/(n_splits+1)))
                    tuning_model = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=tscv.split(X_train))
                else:
                    kfold = StratifiedKFold(n_splits=n_splits)
                    tuning_model = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=kfold)
                tuning_model.fit(X_train, y_train)
                model = tuning_model.best_estimator_
                if verbose:
                    print(tuning_model.best_params_)
                    print(round(tuning_model.best_score_, 2))
            model = model.fit(X_train, y_train, sample_weight=sample_weight_train if use_sample_weight else None)
            
            if model_name == 'CART' and save_trees:
                dt_dir = forecast_dir+time_interval+'/'+ predictors_name+'/'+model_name+'/Decision_Trees/'
                if not os.path.exists(dt_dir):
                    os.makedirs(dt_dir)
                graph = Source(tree.export_graphviz(model, out_file=None, feature_names=[predictor for predictor in predictors], class_names=['Up', 'Down']))
                tree_filename = 'decision_tree_'+str(test_date.year)+'_'+str(test_date.month)+'_'+str(test_date.day)
                # graph.save(filename=tree_filename+'.png', directory=dt_dir)
                graph.render(filename=tree_filename, directory=dt_dir, view=False, format='png')
            if model_name in ['Logistic', 'L2_Logistic', 'EN_Logistic']:
                coefs_dir = forecast_dir+time_interval+'/'+ predictors_name+'/'+model_name+'/Coefs/'
                if not os.path.exists(coefs_dir):
                    os.makedirs(coefs_dir)
                coefs_df.loc[test_date] = list(model.coef_[0]) + list(model.intercept_)
                coefs_df.to_csv(coefs_dir+predictors_name+'_coefs.csv')
                     
        model_forecast = model.predict(cur_X_test)[0]
        model_forecast_prob = model.predict_proba(cur_X_test)[0,1]
        y_pred = model.predict(X_train)
        y_pred_probs = model.predict_proba(X_train)
        
        forecast_df.loc[test_date] = [model_forecast, 
                                      model_forecast_prob,
                                      model.score(X_train, y_train),
                                      precision_score(y_train.values, y_pred),
                                      recall_score(y_train.values, y_pred),
                                      npv(y_train.values, y_pred),
                                      roc_auc_score(y_train.values, y_pred_probs[:, 1]),
                                      f1_score(y_train.values, y_pred)]
        X_train.loc[test_date] = X_test.loc[test_date].copy()
        y_train.loc[test_date] = y_test.loc[test_date]
        sample_weight_train.loc[test_date] = sample_weight_test.loc[test_date]
        if not expanding:
            X_train = X_train.iloc[1:]
            y_train = y_train.iloc[1:]
            sample_weight_train = sample_weight_train.iloc[1:]
        retrain_counter += 1
        retune_counter += 1
        
    # if model_name in ['Logistic', 'L2_Logistic', 'EN_Logistic']:
        # coefs_df
        # plt.figure(figsize=())
        # plt.savefig(coefs_dir+)
        # plt.close()
    
    return forecast_df

def get_oos_metrics(df, forecast_df, test_start_date, predictand, is_rolling=True):
    prev_test_start_date = get_prev_months_first_date(test_start_date)[0] if not is_rolling else test_start_date
    y_test = df[df.index >= test_start_date].copy()[predictand].dropna()
    forecasts = forecast_df.loc[y_test.index, 'Forecast'].values
    forecast_probs = forecast_df.loc[y_test.index, 'Prob'].values
    metrics = pd.Series({'Accuracy' : np.mean((y_test.values == forecasts)), 
                         'Precision' : precision_score(y_test.values, forecasts), 
                         'Recall' : recall_score(y_test.values, forecasts),
                         'NPV' : npv(y_test.values, forecasts),
                         'AUC' : roc_auc_score(y_test.values, forecast_probs),
                         'F1' : f1_score(y_test.values, forecasts),
                         'Prop_Up' : sum(forecasts == 1) / len(forecasts)})
    return metrics

def run_mkt_forecast_models(assets_set, model_name, time_interval, predictand, predictors_name, test_start_date, 
                            retrain_freq=1, retune_freq=1, n_splits=10, param_grid=None, scoring='roc_auc',
                            save_trees=False, expanding=False, use_sample_weight=False, use_ts_split=True, verbose=False, variant_filename=''):
    start_time = time.time()
    # Set up directories
    forecast_dir = '../Portfolios/'+assets_set+'/Mkt_Forecast/'
    metric_dir = forecast_dir+'/'+time_interval+'/'+predictors_name+'/'+model_name+'/Forecast_Metrics/'
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)
    forecast_predictand_dir = forecast_dir+time_interval+'/'+predictors_name+'/'+model_name+'/Forecasts/'+predictand+'/'
    if not os.path.exists(forecast_predictand_dir):
        os.makedirs(forecast_predictand_dir)
    
    # Prep data
    df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_'+time_interval.lower()+'.csv', index_col=0, parse_dates=True).sort_index()
    df[predictand] = df[predictand].astype("category")
    
    ef_coefs = ['r_MVP', 'sigma_MVP', 'u']
    predictors_by_name = {'EF_Coefs'        : ef_coefs, 
                          'EF_Coefs_Lag'    : ef_coefs + [ef_coef+'_1lag' for ef_coef in ef_coefs], 
                          'Tech_Indicators' : ['Sto_K', 'Williams_R', 'Momentum', 'RSI', 'CCI'], 
                          'FF_Factors'      : ['Mkt-RF', 'SMB', 'HML']}
    
    forecast_df = rolling_binary_forecast(assets_set, model_name, predictand, predictors_by_name[predictors_name], df, 
                                          predictors_name, time_interval, test_start_date, retrain_freq=retrain_freq, 
                                          retune_freq=retune_freq, n_splits=n_splits, param_grid=param_grid, scoring=scoring,
                                          save_trees=save_trees, expanding=expanding, use_sample_weight=use_sample_weight, 
                                          use_ts_split=use_ts_split, verbose=verbose)
    forecast_df.to_csv(forecast_predictand_dir + predictand+'_'+model_name+'_'+predictors_name+'_forecasts.csv')
    
    # Calc metrics
    metrics = get_oos_metrics(df, forecast_df, test_start_date, predictand, is_rolling=(time_interval[:7] == 'Rolling'))
    metrics_df = pd.DataFrame(columns=['Time_Interval', 'Predictand', 'Predictor', 'Model', 'Year'] + list(metrics.index))
    metrics_df.loc[len(metrics_df.index)] = [time_interval, predictand, predictors_name, model_name, 'full'] + list(metrics.values)
    # for year in forecast_df.index.year.unique():
    #     cur_df, cur_forecast_df = df[df.index.year == year], forecast_df[forecast_df.index.year == year]
    #     cur_metrics = get_oos_metrics(cur_df, cur_forecast_df, test_start_date, predictand)
    #     metrics_df.loc[len(metrics_df.index)] = [time_interval, predictand, predictors_name, model_name, year] + list(cur_metrics.values)
    metrics_df.to_csv(metric_dir+'binary_mkt_forecast_oos_metrics'+variant_filename+'.csv', index=False)
    
    print('Total Forecast Models Runtime:', str(round((time.time() - start_time)/(60))), 'mins')
    return metrics_df
