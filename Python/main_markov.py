import pandas as pd

from research.markov_markowitz.visualizations.plot_clustering_full import plot_clustering_models_full
from research.markov_markowitz.visualizations.graph_markov_model_full import graph_markov_model_full
from research.markov_markowitz.markov_markowitz_model import markov_markowitz_model


run_full_clustering   = 1
run_full_markov_model = 1
run_in_sample         = 1
run_out_sample        = 1

assets_sets = ['GVMC', 'Sectors', 'Dev_Mkts', 'Combined', 'GVMC_08', 'Dev_Mkts_08']
assets_sets_settings_df = pd.read_csv('asset_sets_settings.csv', index_col=0, parse_dates=['Start_Date', 'Test_Start_Date', 'End_Date'])
assets_sets_settings_df['Assets'] = [assets.split(' ') for assets in assets_sets_settings_df['Assets']]

time_intervals = ['Monthly']
time_interval = 'Monthly'
clustering_models = ['Hierarchical_DTW_Ward']
num_clusters_list = [2, 3, 4, 5]
cluster_features = ['r_MVP', 'sigma_MVP', 'u']
version_name = ''

subfolder_name = 'Constraint_Lev_1.5'


for assets_set in assets_sets:
    test_start_date = assets_sets_settings_df.loc[assets_set, 'Test_Start_Date']
    if run_full_clustering:
        plot_clustering_models_full(assets_set, cluster_features, version_name, clustering_models, num_clusters_list, time_intervals, test_start_date=test_start_date)
    if run_full_markov_model:
        graph_markov_model_full(assets_set, cluster_features, version_name, clustering_models, num_clusters_list, time_intervals)
    if run_in_sample:
        markov_markowitz_model(assets_set, subfolder_name, cluster_features, clustering_models, num_clusters_list, time_interval, test_start_date,
                               weights_sum_list=[1, 0], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint', 
                               weights_agg_method='state-weights', plot_spx_benchmark=(assets_set not in ['Developed_Markets', 'Dev_Mkt', 'Dev_Mkts_08']), 
                               in_sample=True, vol_target=True) # 'prob_weighted_params'
    if run_out_sample:
        markov_markowitz_model(assets_set, subfolder_name, cluster_features, clustering_models, num_clusters_list, time_interval, test_start_date,
                               weights_sum_list=[1, 0], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint', 
                               weights_agg_method='state-weights', plot_spx_benchmark=(assets_set not in ['Developed_Markets', 'Dev_Mkts', 'Dev_Mkts_08']), 
                               in_sample=False, vol_target=True, mult_plot_num_clusters_list=[3, 4, 5]) # 'prob_weighted_params'
    
        
