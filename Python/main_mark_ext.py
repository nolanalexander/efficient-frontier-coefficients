import pandas as pd
from research.markowitz_ext.setup_markowitz_ext_forecast import create_coefs_regr_data
from research.markowitz_ext.visualizations.exploratory_analysis import exploratory_analysis
from research.markowitz_ext.markowitz_ext_forecast import rolling_ef_forecast_regr, rolling_indiv_forecast_regr
from research.markowitz_ext.markowitz_ext_weights_extrac import extract_all_weights
from research.markowitz_ext.simulate_mark_ext_portfolios import simulate_mark_ext_portfolios


run_setup_regr     = 1
run_expl_analysis  = 1
run_indiv_regr     = 1
run_forecast_regr  = 1
run_weights_extrac = 1
run_plot_sim_port  = 1

assets_sets = ['GVMC_and_Bonds', 'Sectors_and_Bonds']
assets_sets_settings_df = pd.read_csv('asset_sets_settings.csv', index_col=0, parse_dates=['Start_Date', 'Test_Start_Date', 'End_Date'])
assets_sets_settings_df['Assets'] = [assets.split(' ') for assets in assets_sets_settings_df['Assets']]

subfolder_name = 'Constraint_Lev_1.5'
lookback_method, window = 'Rolling_1mo', 21


for assets_set in assets_sets:
    assets, start_date, test_start_date, end_date = assets_sets_settings_df.loc[assets_set]
    
    ### Set up Regr Data
    if run_setup_regr:
        create_coefs_regr_data(assets_set, lookback_method, lookback_period=window)
    
    ### Regr Exploratory Analysis
    if run_expl_analysis:
        exploratory_analysis(assets_set, lookback_method)
    
    ### Individual asset return regression
    if run_indiv_regr:
        rolling_indiv_forecast_regr(assets_set, subfolder_name, lookback_method, 
                                    start_date=test_start_date, lookback_period=window, sliding=False)
        
    ### Rolling Forecast Regr
    if run_forecast_regr:
        
        # predictors_by_predictand = { 
        #     'r_MVP_fwd1mo' : ['u_1yr', 'sigma_MVP_1yr'],
        #     'sigma_MVP_fwd1mo' : ['sigma_MVP_1mo', 'Chg_1mo_MA'],
        #     'u_fwd1mo' : ['Chg_3mo_MA'],
        #     }
        
        # predictors_by_predictand = { 
        #     'r_MVP_fwd1mo' : ['u_3mo', 'r_MVP_1mo', 'r_MVP_1yr', 'HML_1mo_MA', 'SMB_1yr_MA'],
        #     'sigma_MVP_fwd1mo' : ['sigma_MVP_1mo', 'sigma_MVP_1yr', 'r_MVP_3mo', 'r_MVP_1yr'],
        #     'u_fwd1mo' : ['Chg_3mo_MA', 'u_1mo', 'u_1yr', 'sigma_MVP_1yr', 'HML_3mo_MA'],
        #     }
        
        predictors_by_predictand = { 
            'r_MVP_fwd1mo'     : ['r_MVP_1yr'], # +'u_1yr'
            'sigma_MVP_fwd1mo' : ['sigma_MVP_1yr'], # +'Chg_1yr_MA'
            'u_fwd1mo'         : ['sigma_MVP_1yr', 'Chg_1yr_MA'],  # +'Chg_1yr_MA', 'u_1yr'
            }
        
        
        
        rolling_ef_forecast_regr(assets_set, subfolder_name, lookback_method, predictors_by_predictand, 
                                 lookback_period=window, model_name='ols', start_date=test_start_date, 
                                 online=True, sliding=False)
    ### Weights Extraction
    if run_weights_extrac:
        extract_all_weights(assets_set, subfolder_name, lookback_method.lower(), 
                            window=window, test_start_date=test_start_date, hyperparameter=None,
                            lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='scaling', smooth_forecasts=False)
    
    ### Plot Simulated Portfolios
    if run_plot_sim_port:
        simulate_mark_ext_portfolios(assets_set, subfolder_name, lookback_method.lower(), test_start_date, transaction_cost=0.01, 
                                     leverage=(2 if assets_set in ['Sectors_and_Bonds'] else 1))


