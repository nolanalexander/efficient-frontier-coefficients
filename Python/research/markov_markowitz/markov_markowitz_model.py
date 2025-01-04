from research.markov_markowitz.prep_markov_markowitz_sim import prep_markov_markowitz_sim
from research.markov_markowitz.simulate_markov_markowitz import simulate_markov_markowitz, aggregate_markov_markowitz_metrics, plot_markov_markowitz, plot_mult_markov_markowitz


def markov_markowitz_model(assets_set, subfolder_name, cluster_features, clustering_models, num_clusters_list, time_interval, test_start_date,
                          lookback=None, weights_sum_list=[-1, 0, 1], lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint', 
                          in_sample=False, retrain_freq=1, weights_agg_method='state-weights', plot_spx_benchmark=True, vol_target=True, mult_plot_num_clusters_list=None):
    # prep_markov_markowitz_sim(assets_set, subfolder_name, cluster_features, clustering_models, num_clusters_list, time_interval, test_start_date,
    #                           lookback=lookback, weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound, max_leverage=max_leverage, max_leverage_method=max_leverage_method, 
    #                           in_sample=in_sample, retrain_freq=retrain_freq, weights_agg_method=weights_agg_method)
    # simulate_markov_markowitz(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, test_start_date, in_sample=in_sample)
    # aggregate_markov_markowitz_metrics(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, in_sample=in_sample)
    # if not in_sample:
    #     plot_markov_markowitz(assets_set, subfolder_name, clustering_models, num_clusters_list, time_interval, test_start_date, plot_spx_benchmark=plot_spx_benchmark, vol_target=vol_target)
    if mult_plot_num_clusters_list is not None:
        plot_mult_markov_markowitz(assets_set, subfolder_name, clustering_models, mult_plot_num_clusters_list, time_interval, test_start_date, plot_spx_benchmark=plot_spx_benchmark, vol_target=vol_target)