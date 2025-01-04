import pandas as pd
import numpy as np
from research.mkt_forecast_markowitz.create_mkt_forecast_data import create_mkt_forecast_data
from research.mkt_forecast_markowitz.visualizations.forecast_data_viz import create_forecast_data_viz
from research.mkt_forecast_markowitz.forecast_mkt import run_mkt_forecast_models
from research.mkt_forecast_markowitz.simple_forecast_backtest import run_simple_forecast_backtest
from research.mkt_forecast_markowitz.conditional_expectation_portfolio import find_all_mkt_forecast_mark_tan_port_weights
from research.mkt_forecast_markowitz.simulate_forecast_mkt_portfolio import simulate_forecast_mkt_portfolio
from forecast_backtest.forecast_metrics import min_ppv_npv, hmean_ppv_npv, ppv, npv
from sklearn.metrics import precision_score, make_scorer


min_ppv_npv_scorer = make_scorer(min_ppv_npv, greater_is_better=True)
hmean_ppv_npv_scorer = make_scorer(hmean_ppv_npv, greater_is_better=True)
ppv_scorer = make_scorer(ppv, greater_is_better=True)
npv_scorer = make_scorer(npv, greater_is_better=True)

run_create_forecast_data = 1
run_forecast_data_viz    = 1
run_forecast_mkt         = 1
run_simple_backtest      = 1
run_cond_exp_port        = 1
run_portfolio_sim        = 1

assets_sets = ['Sectors']
assets_sets_settings_df = pd.read_csv('asset_sets_settings.csv', index_col=0, parse_dates=['Start_Date', 'Test_Start_Date', 'End_Date'])
assets_sets_settings_df['Assets'] = [assets.split(' ') for assets in assets_sets_settings_df['Assets']]

subfolder_name = 'Constraint_Lev_1.5'
time_interval = 'Monthly'
retrain_freq_by_time_interval = {'Rolling_1mo' : 21, 'Monthly' : 1}
retune_freq_by_time_interval = {'Rolling_1mo' : 252, 'Monthly' : 12}
window = 21
forecast_model_name = 'CART'
# forecast_model_name = 'EN_Logistic'
predictand = 'Mkt_Next_Up'
predictors_names = ['EF_Coefs'] # 'Tech_Indicators', 'FF_Factors']

param_grid = {"max_features"    : [None],
              "max_depth"       : np.arange(1, 2+1, 1),
              }

# param_grid = {"C"        : [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#               "l1_ratio" : np.arange(0.1, 0.91, 0.2)
#               }

for assets_set in assets_sets:
    start_date, end_date, test_start_date = assets_sets_settings_df.loc[assets_set, ['Start_Date', 'End_Date', 'Test_Start_Date']]
    
    if run_create_forecast_data:
        create_mkt_forecast_data(assets_set, start_date, end_date)
    if run_forecast_data_viz:
        create_forecast_data_viz(assets_set)
    for predictors_name in predictors_names:
        if run_forecast_mkt:
                run_mkt_forecast_models(assets_set, forecast_model_name, time_interval, predictand, predictors_name, test_start_date, 
                                        retrain_freq=retrain_freq_by_time_interval[time_interval], retune_freq=retune_freq_by_time_interval[time_interval],
                                        n_splits=5, param_grid=param_grid, save_trees=(predictors_name == 'EF_Coefs'),
                                        expanding=1, use_sample_weight=1, use_ts_split=1, scoring='accuracy', verbose=False) # 'roc_auc')# ppv_scorer)

        if run_simple_backtest:
            run_simple_forecast_backtest(assets_set, subfolder_name, time_interval, forecast_model_name, predictand, predictors_name, test_start_date)
        
        if run_cond_exp_port:
            monthly_mkt_forecast_weights_df = find_all_mkt_forecast_mark_tan_port_weights(assets_set, subfolder_name, time_interval, forecast_model_name, predictand, predictors_name, test_start_date, 
                                                                                          lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint', window=None, 
                                                                                          discretize_prob=True, weight_capm=True, up_forecast=(predictand != 'Mkt_Next_Down'))
        if run_portfolio_sim:
            simulate_forecast_mkt_portfolio(assets_set, subfolder_name, time_interval, forecast_model_name, predictors_name, test_start_date, max_leverage=1.5)
        
