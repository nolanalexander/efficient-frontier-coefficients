import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import time
from tqdm import tqdm
import os

'''
Runs a rolling regression to forecast EF coefs
'''

def get_regr_data(asset_set, lookback_method, dropnas_in_any_col=False):
    coefs_regr_df = pd.read_csv('../Portfolios/'+asset_set+'/Assets_Data/coefs_regr_data_'+lookback_method.lower()+'.csv', index_col=0)
    coefs_regr_df.index = pd.to_datetime(coefs_regr_df.index)
    if(dropnas_in_any_col):
        num_init_rows = len(coefs_regr_df.index)
        coefs_regr_df = coefs_regr_df.dropna()
        print('Dropping NAs resulted in going from', num_init_rows, 'rows to', len(coefs_regr_df.index), 'rows')
    coefs = ['r_MVP', 'sigma_MVP', 'u']
    lookback_name_by_method = {'rolling_1mo' : '1mo', 
                               'rolling_3mo' : '3mo', 
                               'yearly' : '1yr',
                               'quarterly' : '3mo'}
    predictand_fwd_name = lookback_name_by_method[lookback_method.lower()]
    predictands = [coef + '_fwd' + predictand_fwd_name for coef in coefs]
    non_predictors = ['Chg_1D_Fwd', 'Date', 'Date_coefs', 'Date_preds', 'Year', 'Year_coefs', 
                      'Year_preds', 'Quarter_coefs', 'Quarter_preds', 'Quarter'] + predictands
    all_predictors = coefs_regr_df.columns.values[~coefs_regr_df.columns.isin(non_predictors)]
    return coefs_regr_df, all_predictors, predictands

def transform_to_interaction(df, degree=2):
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias = False)
    df_interac = poly.fit_transform(df)
    df_interac_names = np.array(poly.get_feature_names(df.columns))
    df_interac = pd.DataFrame(df_interac, columns=df_interac_names)
    df_interac.index = df.index
    return df_interac

def prep_data(train_df, test_df, predictors, interactions=False):
    X_train, X_test = train_df[predictors].copy(), test_df[predictors].copy()
    if(interactions):
        X_train = transform_to_interaction(X_train, 2)
        X_test = transform_to_interaction(X_test, 2)
    return X_train, X_test

def r2(y, y_pred):
    return pd.DataFrame({'y' : y, 'y_pred' : y_pred}).corr().iloc[0,1]**2

def r2_cv(model, X, y):
    y_pred = model.predict(X)
    return r2(y, y_pred)

def fit_forecast_model(train_df, test_df, predictand, predictors, model, verbose=False, interactions=False, num_folds=10):
    X_train, X_test = prep_data(train_df, test_df, predictors, interactions=interactions)
    y_train = train_df[predictand].copy()
    forecaster = model.fit(X_train, y_train)
    forecaster_r2 = forecaster.score(X_train, y_train)
    forecaster_cv_r2 = np.mean(cross_val_score(forecaster, X_train, y_train, cv=num_folds, scoring=r2_cv))
    forecaster_params = X_train.columns[np.nonzero(np.array(forecaster.coef_))].values
    if verbose:
        print(forecaster_params)
        print(predictand + ' model Num Params: ' + str(len(forecaster_params)))
        print(predictand + ' model R2: ' + str(forecaster_r2))
        print(predictand + ' model CV R2: ' + str(forecaster_cv_r2))
        print(sm.OLS(y_train, sm.add_constant(X_train)).fit().summary())
    predictand_forecast = forecaster.predict(X_test)
    return pd.Series([predictand_forecast, forecaster_r2, forecaster_cv_r2, len(forecaster_params)], index=['forecast', 'R2', 'CV_R2', 'num_params'])

def forecast_EF_coefs(train_df, test_df, predictors_by_predictand, 
                      model_name='ols', num_folds=10, interactions=False, quiet=True):
    
    model_by_name = {'lasso' : LassoCV(cv=num_folds, n_alphas=10**3, max_iter=10**6),
                     'ols'   : LinearRegression()}
    model = model_by_name[model_name]
    
    # Forecast coefs
    predictands = [predictand for predictand in predictors_by_predictand.keys()]
    r_MVP_forecast_res = fit_forecast_model(train_df, test_df, predictands[0], predictors_by_predictand[predictands[0]], model)
    sigma_MVP_forecast_res = fit_forecast_model(train_df, test_df, predictands[1], predictors_by_predictand[predictands[1]], model)
    u_forecast_res = fit_forecast_model(train_df, test_df, predictands[2], predictors_by_predictand[predictands[2]], model)
    
    # Organize results
    if predictands[0][-10:] == '_daily_chg':
        cur_r_MVP = test_df[predictands[0][:-10]].iloc[0]
        cur_sigma_MVP = test_df[predictands[1][:-10]].iloc[0]
        cur_u = test_df[predictands[2][:-10]].iloc[0]
        forecast_coefs_df = pd.DataFrame({predictands[0] : cur_r_MVP +  cur_r_MVP * r_MVP_forecast_res['forecast'], 
                                          predictands[1] : cur_sigma_MVP + cur_sigma_MVP * sigma_MVP_forecast_res['forecast'], 
                                          predictands[2] : cur_u + cur_u * u_forecast_res['forecast'] })
    else:
        forecast_coefs_df = pd.DataFrame({predictands[0] : r_MVP_forecast_res['forecast'], 
                                          predictands[1] : sigma_MVP_forecast_res['forecast'], 
                                          predictands[2] : u_forecast_res['forecast'] })
    forecast_coefs_df.index = test_df.index
    results_df = pd.DataFrame(columns=predictands)
    results_df.loc['R2'] = r_MVP_forecast_res['R2'], sigma_MVP_forecast_res['R2'], u_forecast_res['R2']
    results_df.loc['CV_R2'] = r_MVP_forecast_res['CV_R2'], sigma_MVP_forecast_res['CV_R2'], u_forecast_res['CV_R2']
    results_df.loc['Num_Params'] = r_MVP_forecast_res['num_params'], sigma_MVP_forecast_res['num_params'], u_forecast_res['num_params']
    return forecast_coefs_df, results_df
    
def rolling_ef_forecast_regr(asset_set, subfolder_name, lookback_method, predictors_by_predictand, start_date,
                                 model_name='ols', lookback_period=int(252/4), online=True, sliding=False, quiet=True):
    start_time = time.time()
    coefs_regr_df, predictors, predictands = get_regr_data(asset_set, lookback_method)
    predictands = [predictand for predictand in predictors_by_predictand.keys()]
    all_predictors = coefs_regr_df.columns.values[~coefs_regr_df.columns.isin(['Date_coefs', 'Date_preds', 'Chg_1D_Fwd', 'Quarter', 'Year', 'A', 'B', 'C'] + predictands)].tolist()
    ef_forecast_coefs_df = pd.DataFrame(columns=predictands)
    cv_R2_df = pd.DataFrame(columns=predictands)
    R2_df = pd.DataFrame(columns=predictands)
    
    train_df = coefs_regr_df.loc[coefs_regr_df.index < start_date].dropna()# .iloc[:-lookback_period].copy()
    # store_df = coefs_regr_df.loc[coefs_regr_df.index < start_date].iloc[-lookback_period:].copy() # Data you do not have the fwd n-month coefs for yet
    test_df = coefs_regr_df.loc[coefs_regr_df.index >= start_date].copy()
    train_window = len(train_df.index)
    
    p_bar = tqdm(test_df.index)
    for test_date in p_bar:
        p_bar.set_description(f'{pd.to_datetime(test_date).date()} ')
        
        # Forecast EF coefs
        forecast_input = test_df.loc[[test_date], all_predictors]
        cur_ef_forecast_coefs_df, cur_ef_forecast_results_df = forecast_EF_coefs(train_df, forecast_input, 
                                                                                  predictors_by_predictand, model_name=model_name,
                                                                                  num_folds=10, interactions=False, quiet=True)
        # print(cur_ef_forecast_coefs_df)
        ef_forecast_coefs_df = pd.concat([ef_forecast_coefs_df, cur_ef_forecast_coefs_df])
        cv_R2_df.loc[test_df.index[0]] = cur_ef_forecast_results_df.loc['CV_R2'].values
        R2_df.loc[test_df.index[0]] = cur_ef_forecast_results_df.loc['R2'].values
        
        # Update rolling window
        if online:
            train_df.loc[test_date] = test_df.loc[test_date]
        # train_df = pd.concat([train_df, store_df.iloc[[0]]])
        # store_df = pd.concat([store_df, test_df.iloc[[0]]])
        # print(train_df)
        if(sliding):
            train_df = train_df.iloc[-train_window:]
        
        # store_df = store_df.iloc[1:]
        test_df = test_df.iloc[1:]
        # print(train_df.index)
        # print(test_df.index)
    
    # def r2_score(predictors, predictand):
    #     return LinearRegression().fit(coefs_regr_df.dropna()[predictors], coefs_regr_df.dropna()[predictand]).score(coefs_regr_df.dropna()[predictors], coefs_regr_df.dropna()[predictand])
    
    # forecast_input = coefs_regr_df.dropna()[all_predictors]
    # for col in forecast_input.columns:
    #     print(col)
    #     predictors_by_predictand = { 
    #         'r_MVP_1mo_daily_chg_fwd' : [col], # , 'Chg_1mo_MA'],
    #         'sigma_MVP_1mo_daily_chg_fwd' : [col] ,# , 'Chg_1mo_MA'],
    #         'u_1mo_daily_chg_fwd' : [col],
    #         }
        
    #     cur_ef_forecast_coefs_df, cur_ef_forecast_results_df = forecast_EF_coefs(coefs_regr_df.dropna(), forecast_input, 
    #                                                                               predictors_by_predictand, model_name=model_name,
    #                                                                               num_folds=10, interactions=False, quiet=True)
    #     print(cur_ef_forecast_results_df.loc[['R2', 'CV_R2']].round(3))
        
    # predictors_by_predictand = { 
    #     'r_MVP_fwd1mo' : ['sigma_MVP_1yr', 'r_MVP_3mo', 'u_1yr'],# , 'Chg_1mo_MA'],
    #     'sigma_MVP_fwd1mo' : ['sigma_MVP_1yr', 'Chg_1mo_MA'],
    #     'u_fwd1mo' : ['sigma_MVP_1yr', 'u_1yr', 'Chg_3mo_MA']
    #     }
    
    # cur_ef_forecast_coefs_df, cur_ef_forecast_results_df = forecast_EF_coefs(coefs_regr_df.dropna(), forecast_input, 
    #                                                                           predictors_by_predictand, model_name=model_name,
    #                                                                           num_folds=10, interactions=False, quiet=True)
    # print(cur_ef_forecast_results_df.loc[['R2', 'CV_R2']].round(3))
    
    test_df = coefs_regr_df.loc[coefs_regr_df.index >= start_date].copy()
    ef_forecast_coefs_df = ef_forecast_coefs_df.astype(float)
    oos_R2_df = pd.DataFrame({predictands[0] : [r2(test_df[predictands[0]], ef_forecast_coefs_df[predictands[0]])], 
                              predictands[1] : [r2(test_df[predictands[1]], ef_forecast_coefs_df[predictands[1]])], 
                              predictands[2] : [r2(test_df[predictands[2]], ef_forecast_coefs_df[predictands[2]])] })
  
    lookback_method = lookback_method.lower()
    me_dir = '../Portfolios/'+asset_set+'/Markowitz_Ext/'
    ef_forecast_coefs_df.to_csv(me_dir+'forecasted_EF_coefs_'+lookback_method+'.csv')
    R2_df.to_csv(me_dir+'EF_R2s_'+lookback_method+'.csv')
    cv_R2_df.to_csv(me_dir+'EF_CV_R2s_'+lookback_method+'.csv')
    oos_R2_df.to_csv(me_dir+'EF_OoS_R2s_'+lookback_method+'.csv', index=False)
    if not quiet:
        print('R2_DF')
        print(R2_df.mean())
        print('CV_R2_DF')
        print(cv_R2_df.mean())
    print('Rolling forecast runtime:', str(round((time.time() - start_time)/60, 1)),'mins')
    
    return ef_forecast_coefs_df, cv_R2_df

def forecast_indiv(train_df, test_df, ticker):
    train_df, test_df = train_df.copy(), test_df.copy()
    model = LinearRegression()
    
    # Forecast coefs
    forecast_res = fit_forecast_model(train_df, test_df, ticker+'_Chg_MA_fwd', [ticker+'_Chg_MA'], model)
    forecast_w_eq_res = fit_forecast_model(train_df, test_df, ticker+'_Chg_MA_fwd', [ticker+'_Chg_MA', 'Eq_Weight_MA'], model)
    
    # Organize results
    forecast_df = pd.DataFrame({'AR' : forecast_res['forecast'],
                                'AR_and_Mkt' : forecast_w_eq_res['forecast']})
    forecast_df.index = test_df.index
    results_df = pd.DataFrame(columns=forecast_df.columns)
    results_df.loc['R2'] = forecast_res['R2'], forecast_w_eq_res['R2']
    results_df.loc['CV_R2'] = forecast_res['CV_R2'], forecast_w_eq_res['CV_R2']
    return forecast_df, results_df

def rolling_indiv_forecast_regr(assets_set, subfolder_name, lookback_method, start_date,
                                lookback_period=int(252/4), sliding=False, quiet=True):
    start_time = time.time()
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    
    
    forecast_df_ar = pd.DataFrame(columns=assets_returns_df.columns)
    cv_R2_df_ar = pd.DataFrame(columns=assets_returns_df.columns)
    R2_df_ar = pd.DataFrame(columns=assets_returns_df.columns)
    forecast_df_ar_and_mkt = pd.DataFrame(columns=assets_returns_df.columns)
    cv_R2_df_ar_and_mkt = pd.DataFrame(columns=assets_returns_df.columns)
    R2_df_ar_and_mkt = pd.DataFrame(columns=assets_returns_df.columns)
    oos_R2_df = pd.DataFrame(columns=assets_returns_df.columns, index=['AR', 'AR_and_Mkt'])
    
    p_bar = tqdm(assets_returns_df.columns)
    for ticker in p_bar:
        p_bar.set_description('ticker ')
        regr_df = pd.DataFrame({ticker+'_Chg_MA' : assets_returns_df[ticker].rolling(lookback_period).mean(), 
                                'Eq_Weight_MA' : (assets_returns_df.sum(1)/len(assets_returns_df.columns)).rolling(lookback_period).mean()
                                })
        regr_df[ticker+'_Chg_MA_fwd'] = regr_df[ticker+'_Chg_MA'].shift(-lookback_period)
        regr_df = regr_df.dropna()
        train_df = regr_df.loc[regr_df.index < start_date] #.iloc[:-lookback_period].copy()
        # store_df = regr_df.loc[regr_df.index < start_date].iloc[-lookback_period:].copy() # Data you do not have the fwd n-month coefs for yet
        test_df = regr_df.loc[regr_df.index >= start_date].copy()
        train_window = len(train_df.index)
    
        for test_date in test_df.index:
            
            # Forecast EF coefs
            forecast_input = test_df.loc[[test_date], :]
            cur_forecast_df, cur_forecast_results_df = forecast_indiv(train_df, forecast_input, ticker)
            forecast_df_ar.loc[test_date, ticker] = cur_forecast_df['AR'].iloc[0]
            cv_R2_df_ar.loc[test_df.index[0], ticker] = cur_forecast_results_df.loc['CV_R2', 'AR']
            R2_df_ar.loc[test_df.index[0], ticker] = cur_forecast_results_df.loc['R2', 'AR']
            forecast_df_ar_and_mkt.loc[test_df.index[0], ticker] = cur_forecast_df['AR_and_Mkt'].iloc[0]
            cv_R2_df_ar_and_mkt.loc[test_df.index[0], ticker] = cur_forecast_results_df.loc['CV_R2', 'AR_and_Mkt']
            R2_df_ar_and_mkt.loc[test_df.index[0], ticker] = cur_forecast_results_df.loc['R2', 'AR_and_Mkt']
            
            # Update rolling window
            train_df = pd.concat([train_df, test_df.iloc[[0]]])
            # train_df = pd.concat([train_df, store_df.iloc[[0]]])
            # store_df = pd.concat([store_df, test_df.iloc[[0]]])
            # print(train_df)
            if(sliding):
                train_df = train_df.iloc[-train_window:]
            
            # store_df = store_df.iloc[1:]
            test_df = test_df.iloc[1:]
            # print(train_df.index)
            # print(test_df.index)
    
        test_df = regr_df.loc[regr_df.index >= start_date].copy()
        forecast_df_ar = forecast_df_ar.astype(float)
        forecast_df_ar_and_mkt = forecast_df_ar_and_mkt.astype(float)
        oos_R2_df[ticker] = [r2(test_df[ticker+'_Chg_MA_fwd'], forecast_df_ar[ticker]),
                             r2(test_df[ticker+'_Chg_MA_fwd'], forecast_df_ar_and_mkt[ticker])]
    
    avg_cv_R2_df = pd.DataFrame({'AR' : cv_R2_df_ar.mean(), 'AR_and_Mkt' : cv_R2_df_ar_and_mkt.mean()})
  
    lookback_method = lookback_method.lower()
    me_indiv_dir = '../Portfolios/'+assets_set+'/Markowitz_Ext/indiv/'
    if not os.path.exists(me_indiv_dir):
        os.makedirs(me_indiv_dir)
    forecast_df_ar.to_csv(me_indiv_dir+'forecasted_indiv_ar_'+lookback_method+'.csv')
    forecast_df_ar_and_mkt.to_csv(me_indiv_dir+'forecasted_indiv_ar_and_mkt_'+lookback_method+'.csv')
    R2_df_ar.to_csv(me_indiv_dir+'indiv_ar_R2s_'+lookback_method+'.csv')
    R2_df_ar_and_mkt.to_csv(me_indiv_dir+'indiv_ar_and_mkt_R2s_'+lookback_method+'.csv')
    cv_R2_df_ar.to_csv(me_indiv_dir+'indiv_ar_CV_R2s'+lookback_method+'.csv')
    cv_R2_df_ar_and_mkt.to_csv(me_indiv_dir+'indiv_ar_and_mkt_CV_R2s'+lookback_method+'.csv')
    oos_R2_df.to_csv(me_indiv_dir+'indiv_OoS_R2s_'+lookback_method+'.csv')
    avg_cv_R2_df.to_csv(me_indiv_dir+'indiv_avg_CV_R2s_'+lookback_method+'.csv')
    if not quiet:
        print('R2_DF')
        print(oos_R2_df)
        print('CV_R2_DF')
        print(avg_cv_R2_df)
    print('Rolling forecast runtime:', str(round((time.time() - start_time)/60, 1)),'mins')
    
    return forecast_df_ar, forecast_df_ar_and_mkt, oos_R2_df, avg_cv_R2_df

def rolling_forecast_indiv(asset_set, subfolder_name, lookback_method, predictors_by_predictand, start_date,
                           model_name='ols', lookback_period=int(252/4), sliding=False, quiet=True):
    pass

def rolling_forecast_regr_yearly(asset_set, subfolder_name, lookback_method, predictors_by_predictand, 
                                 model_name='ols', start_year=None):
    pass

def rolling_forecast_regr_quarterly(asset_set, subfolder_name, lookback_method, predictors_by_predictand, 
                                    model_name='ols', start_quarter=None):
    pass
