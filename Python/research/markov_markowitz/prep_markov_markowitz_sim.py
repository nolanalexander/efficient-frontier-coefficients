import pandas as pd
import numpy as np
import os
import time
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from itertools import product

from data_processing.read_in_data import read_in_rf
from data_processing.ts_utils import get_quarters, get_month_first_dates
from portfolio_optimization.mean_estimation import mean
from portfolio_optimization.cov_estimation import cov_shrinkage, cov
from portfolio_optimization.markowitz import est_markowitz_params_and_select_weights, select_EF_portfolio_weights
from research.markov_markowitz.clustering import cluster
from research.markov_markowitz.markov_model import calc_transition_matrix

'''
Prepares the Markov Markowitz simulation:
1) Clusters by EF coefs
2) Calculate state transition probabilities
3) Creates a mapping of states to optimal weights
'''

def calc_tan_port_weights_in_cluster(assets_returns_df, rf, mean_estimator=mean, cov_estimator=cov, mean_est_kwargs={}, cov_est_kwargs={}, lookback=None, 
                                     weights_sum_list=[-1, 0, 1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, date=None):
    assets_returns_df = assets_returns_df.drop(columns=['State'])
    assets_returns_df = assets_returns_df if lookback is None else assets_returns_df.iloc[-lookback:]
    tan_port_weights = est_markowitz_params_and_select_weights(assets_returns_df, rf, portfolio_selection='tangency', 
                                                                mean_estimator=mean_estimator, cov_estimator=cov_estimator, mean_est_kwargs=mean_est_kwargs, cov_est_kwargs=cov_est_kwargs,
                                                                weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, 
                                                                max_leverage_method=max_leverage_method, date=date)
    return pd.Series(tan_port_weights, index=assets_returns_df.columns)

def get_state_trans_prob_weighted_weights(time_period, predictions_by_time_period, 
                                          trans_mat_by_time_period, state_weight_df_by_time_period, in_sample=False):
    time_period = time_period if not in_sample else list(predictions_by_time_period.keys())[0]
    cur_state = predictions_by_time_period[time_period]['State'].iloc[-1]
    state_weight_df = state_weight_df_by_time_period[time_period]
    trans_mat = trans_mat_by_time_period[time_period]
    pdf = trans_mat.loc[cur_state, state_weight_df.index.astype(str)]
    prob_weighted_state_weights = state_weight_df.T.dot(pdf.values)
    return prob_weighted_state_weights

def get_time_interval_mean_sd_df(spx_df, get_mean=True, get_sd=True):
    spx_mean_df = spx_df.groupby(level=0).mean().rename(columns={'Chg' : 'Mkt_Avg_Return'}).copy()
    spx_sd_df = spx_df.groupby(level=0).std().rename(columns={'Chg' : 'Mkt_Return_SD'}).copy()
    if not get_mean and not get_sd:
        raise ValueError('Both get_mean and get_sd are False')
    if get_mean and get_sd:
        spx_mean_sd_df = spx_mean_df.merge(spx_sd_df, left_index=True, right_index=True)
    else:
        spx_mean_sd_df = spx_mean_df if get_mean else spx_sd_df
    return spx_mean_sd_df

def add_parametrics_to_ef_coefs(spx_df, ef_coefs_df_yearly, ef_coefs_df_quarterly, ef_coefs_df_monthly, add_mean=True, add_sd=True):
    spx_yearly_df, spx_quarterly_df, spx_monthly_df = spx_df.copy(), spx_df.copy(), spx_df.copy()
    spx_yearly_df.index = spx_yearly_df.index.year
    spx_quarterly_df.index = get_quarters(spx_quarterly_df.index)
    spx_monthly_df.index = get_month_first_dates(spx_monthly_df.index)
    
    spx_yearly_df = get_time_interval_mean_sd_df(spx_yearly_df, get_mean=add_mean, get_sd=add_sd)
    spx_quarterly_df = get_time_interval_mean_sd_df(spx_quarterly_df, get_mean=add_mean, get_sd=add_sd)
    spx_monthly_df = get_time_interval_mean_sd_df(spx_monthly_df, get_mean=add_mean, get_sd=add_sd)
    
    ef_coefs_df_yearly = ef_coefs_df_yearly.copy().merge(spx_yearly_df, left_index=True, right_index=True)
    ef_coefs_df_quarterly = ef_coefs_df_quarterly.copy().merge(spx_quarterly_df, left_index=True, right_index=True)
    ef_coefs_df_monthly = ef_coefs_df_monthly.copy().merge(spx_monthly_df, left_index=True, right_index=True)
    return ef_coefs_df_yearly, ef_coefs_df_quarterly, ef_coefs_df_monthly

def prep_markov_markowitz_sim(assets_set, subfolder_name, cluster_features, clustering_models, num_clusters_list, time_interval, test_start_date, end_date=None,
                              lookback=None, weights_sum_list=[-1, 0, 1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method=None, in_sample=False, 
                              weights_agg_method='prob_weighted_params', retrain_freq=1):
    markov_dir = '../Portfolios/'+assets_set+'/Markov/Markov_Markowitz/'
    
    start_time = time.time()
    
    # Read in and prepare data
    ef_coefs_df_yearly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_yearly.csv', index_col=0)
    ef_coefs_df_quarterly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_quarterly.csv', index_col=0, parse_dates=True)
    ef_coefs_df_monthly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_monthly.csv', index_col=0, parse_dates=True)
    ef_coefs_df_rolling_1mo = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_rolling_1mo.csv', index_col=0, parse_dates=True)
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    rfs = read_in_rf(assets_returns_df.index) 
    
    # Add parametrics
    # spx_df = read_in_spxtr(assets_set, assets_returns_df.index)[['Chg']]
    # ef_coefs_df_yearly, ef_coefs_df_quarterly, ef_coefs_df_monthly = add_parametrics_to_ef_coefs(spx_df, ef_coefs_df_yearly, ef_coefs_df_quarterly, ef_coefs_df_monthly, add_mean=True, add_sd=False)
    
    ef_coefs_df_by_interval = {'Yearly' : ef_coefs_df_yearly, 'Quarterly' : ef_coefs_df_quarterly, 'Monthly' : ef_coefs_df_monthly, 'Rolling_1mo' : ef_coefs_df_rolling_1mo}
    
    assets = assets_returns_df.columns
    assets_returns_df['Year'] = assets_returns_df.index.year
    assets_returns_df['Quarter'] = get_quarters(assets_returns_df.index)
    assets_returns_df['Month'] = get_month_first_dates(assets_returns_df.index)
    
    weights_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Weights/'
    
    mm_weights_by_time_interval_df = pd.DataFrame(columns=assets)
    silhouette_scores_df = pd.DataFrame(columns=['Clustering_Model', 'Num_Clusters', 'Silhouette_Score'])
    
    # Set up dataframes and indices
    cur_ef_coefs_df = ef_coefs_df_by_interval[time_interval][cluster_features].dropna().copy()
    
    if end_date is not None:
        cur_ef_coefs_df = cur_ef_coefs_df.loc[cur_ef_coefs_df.index <= end_date]
    is_rolling = time_interval[:7] == 'Rolling'
    prior_time_periods = cur_ef_coefs_df.index.to_series().shift(1).iloc[1:] if not is_rolling else cur_ef_coefs_df.index.to_series() # Must shift yearly/monthly back, but not rolling
    prior_test_time_periods = prior_time_periods[prior_time_periods.index >= (test_start_date if not time_interval == 'Yearly' else test_start_date.year)]
    train_dates = cur_ef_coefs_df.index[cur_ef_coefs_df.index < (test_start_date if not time_interval == 'Yearly' else test_start_date.year)]
    prior_test_time_periods = prior_test_time_periods if not in_sample else train_dates.to_series().shift(1).iloc[16+1:] # 4-year start for in-samp
    
    def process_clustering(clustering_model, num_clusters, train_ef_coefs_df):
        predictions, fitted_model = cluster(train_ef_coefs_df, clustering_model, num_clusters, return_model=True)
        predictions = pd.DataFrame({'State' : predictions})
        train_silhouette_score = silhouette_score(train_ef_coefs_df, predictions['State'])
        trans_mat = calc_transition_matrix(predictions['State'])
        return predictions, trans_mat, train_silhouette_score, fitted_model
    
    p_bar = tqdm(list(product(clustering_models, num_clusters_list)))
    for clustering_model, num_clusters in p_bar:

        # Set up directories
        mm_model_dir = markov_dir+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
        cluster_pred_dir = mm_model_dir +'Cluster_Predictions/'
        if not os.path.exists(cluster_pred_dir):
            os.makedirs(cluster_pred_dir)
        trans_mat_dir = mm_model_dir+'Transition_Matrix/'
        if not os.path.exists(trans_mat_dir):
            os.makedirs(trans_mat_dir)
        mm_weights_dir = weights_dir+'Markov_Markowitz/'+time_interval+'/'+clustering_model+'/'+str(num_clusters)+'_Clusters/'
        state_weights_dir = mm_weights_dir + 'State_Weights/'
        if not os.path.exists(state_weights_dir):
            os.makedirs(state_weights_dir)
            
        if in_sample:
            prior_test_time_period = prior_test_time_periods[prior_test_time_periods.index[-1]]
            train_ef_coefs_df = cur_ef_coefs_df[cur_ef_coefs_df.index <= prior_test_time_period]
            predictions, trans_mat, train_silhouette_score, fitted_model = process_clustering(clustering_model, num_clusters, train_ef_coefs_df)
            silhouette_scores_df.loc[len(silhouette_scores_df)] = [clustering_model, num_clusters, train_silhouette_score]
        
        counter = 0
        for cur_test_time_period in prior_test_time_periods.index:
            p_bar.set_description(f'{clustering_model} {num_clusters}, {pd.to_datetime(cur_test_time_period).date()}  ')
            
            prior_test_time_period = prior_test_time_periods[cur_test_time_period]
            train_dates = assets_returns_df.index[assets_returns_df[time_interval[:-2]] <= prior_test_time_period if not is_rolling else assets_returns_df.index <= prior_test_time_period]
            train_assets_returns_df = assets_returns_df.loc[train_dates].copy()
            train_rfs = rfs.loc[train_dates].copy()
            need_to_retrain = counter >= retrain_freq or cur_test_time_period == prior_test_time_periods.index[:2][0]
            if need_to_retrain or in_sample:
                counter = 0
                
                if not in_sample:
                    train_ef_coefs_df = cur_ef_coefs_df[cur_ef_coefs_df.index <= prior_test_time_period].copy()
                    predictions, trans_mat, train_silhouette_score, fitted_model = process_clustering(clustering_model, num_clusters, train_ef_coefs_df)
                    filename_ending = str(prior_test_time_period)+'.csv' if time_interval == 'Yearly' else str(prior_test_time_period.year) +'_'+str(prior_test_time_period.month)+'.csv'
                    predictions.to_csv(cluster_pred_dir + 'cluster_predictions_' + filename_ending)
                    trans_mat.to_csv(trans_mat_dir + 'transition_matrix_' + filename_ending)
                cur_state = predictions.loc[prior_test_time_period, 'State']
            else:
                cur_state = fitted_model.predict(train_ef_coefs_df.iloc[[-1]])[0] + 1
                predictions.loc[prior_test_time_period, 'State'] = cur_state
            if is_rolling:
                train_clustered_assets_returns_df = train_assets_returns_df.merge(predictions, left_index=True, right_index=True).copy()
            else:
                train_clustered_assets_returns_df = train_assets_returns_df.merge(predictions, left_on=time_interval[:-2], right_index=True).copy()
            train_clustered_assets_returns_df = train_clustered_assets_returns_df.drop(columns=['Year', 'Quarter', 'Month'])
            start_time = time.time()
            
            if weights_agg_method == 'state-weights':
                if need_to_retrain:
                    # Calculate and save state weights
                    state_weights_df = train_clustered_assets_returns_df.groupby('State').apply(calc_tan_port_weights_in_cluster, rf=train_rfs.iloc[-1], lookback=lookback,
                                                                                                weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound,
                                                                                                max_leverage=max_leverage, max_leverage_method=max_leverage_method, date=prior_test_time_period)
                    if not in_sample:
                        state_weights_df.to_csv(state_weights_dir + 'state_weights_' + filename_ending)
                
                # Calculates Prob-weighted State Weights
                pdf = trans_mat.loc[cur_state, state_weights_df.index]
                pdf = pdf/pdf.sum() if not np.isclose(pdf.sum(), 1) else pdf
                prob_weighted_state_weights = state_weights_df.T.dot(pdf.values)
                mm_weights_by_time_interval_df.loc[cur_test_time_period] = prob_weighted_state_weights
            
            elif weights_agg_method == 'prob_weighted_params':
                # Calculate Prob-weighted Return and Covariance Implied Weights
                train_states_exp_returns_df = train_clustered_assets_returns_df.groupby('State').mean()
                train_states_cov_mats_df = train_clustered_assets_returns_df.groupby('State').cov()
                pdf = trans_mat.loc[cur_state, train_states_exp_returns_df.index]
                pdf = pdf/pdf.sum() if not np.isclose(pdf.sum(), 1) else pdf
                train_states_prob_weighted_exp_returns = pdf.dot(train_states_exp_returns_df)
                prob_weighted_cov_mats = pdf.to_numpy().reshape((len(train_states_exp_returns_df.index), 1, 1)) * train_states_cov_mats_df.to_numpy().reshape((len(train_states_exp_returns_df.index), len(assets), len(assets)))
                train_states_prob_weighted_cov_mat = pd.DataFrame(prob_weighted_cov_mats.sum(axis=0), columns=assets, index=assets)
                prob_weighted_implied_param_weights = select_EF_portfolio_weights('tangency', train_states_prob_weighted_exp_returns, train_states_prob_weighted_cov_mat, train_rfs.iloc[-1],
                                                                                  weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, 
                                                                                  max_leverage=max_leverage, max_leverage_method=max_leverage_method,
                                                                                  date=prior_test_time_period)
                mm_weights_by_time_interval_df.loc[cur_test_time_period] = prob_weighted_implied_param_weights
            else:
                raise ValueError(f'Invalid weights aggregation method: {weights_agg_method}')
            counter += 1
        if not is_rolling:
            date_time_period_df = assets_returns_df[[time_interval[:-2]]]
            mm_weights = mm_weights_by_time_interval_df.merge(date_time_period_df, left_index=True, right_on=time_interval[:-2]).drop(columns=[time_interval[:-2]])
        else:
            mm_weights = mm_weights_by_time_interval_df.copy()
        filename_ending = time_interval.lower()+'_'+clustering_model+'_'+str(num_clusters)+'_clusters_' + ('insamp' if in_sample else 'outsamp')
        mm_weights.to_csv(mm_weights_dir+'markov_markowitz_weights_'+filename_ending+'.csv')
    if in_sample:
        mm_metrics_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Metrics/Markov_Markowitz/'+time_interval+'/'
        if not os.path.exists(mm_metrics_dir):
            os.makedirs(mm_metrics_dir)
        silhouette_scores_df = silhouette_scores_df.sort_values('Silhouette_Score', ascending=False)
        silhouette_scores_df.to_csv(mm_metrics_dir + 'silhouette_scores_insamp.csv', index=False)
    print('Prep Markov Markowitz Simulations Run Time:', round((time.time() - start_time)/60), 'mins')
                
                
