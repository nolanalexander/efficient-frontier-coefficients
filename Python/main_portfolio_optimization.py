from data_processing.read_in_data import read_in_assets_data
from portfolio_optimization.visualizations.plot_ef_vis import plot_yearly_efs_over_time, plot_ef_tan_port
from portfolio_optimization.portfolio_sets_inventory import process_common_portfolio_set
from portfolio_optimization.assets_selection import assets_selection_suite
import datetime as dt
import pandas as pd

run_assets_selection   = 1
run_read_in_data       = 1
run_process_portfolios = 1
run_plot_ef_viz        = 1

assets_sets = ['GVMC', 'Sectors', 'Dev_Mkts']
# assets_sets = ['Combined', 'Combined_Ex_SP500']
# assets_sets = ['GVMC_08', 'Dev_Mkts_08']
assets_sets_settings_df = pd.read_csv('asset_sets_settings.csv', index_col=0, parse_dates=['Start_Date', 'Test_Start_Date', 'End_Date'])
assets_sets_settings_df['Assets'] = [assets_set.split(' ') for assets_set in assets_sets_settings_df['Assets']]

subfolder_name = 'Constraint_Lev_1.5' # 'Scaling_Lev_1.5'

if run_assets_selection:
    tickers = ['FDGRX', 'FLPSX', 'FOCPX', 'FGRIX', 'OPOCX', 'HRTVX',
               'XLK', 'XLV', 'XLF', 'XLE', 'XLB', 'XLY', 'XLI', 'XLU', 'XLP',
               '^SP500TR', '^HSI', '^GDAXI', '^FCHI', '^GSPTSE']
    start_date = pd.to_datetime('2008-01-01')
    end_date = pd.to_datetime('2022-12-31')
    corr_mat, corr_df  = assets_selection_suite(tickers, start_date, end_date)

for assets_set in assets_sets:
    
    assets, start_date, test_start_date, end_date = assets_sets_settings_df.loc[assets_set]
    
    # Read in assets data
    if(run_read_in_data):
        read_in_assets_data(assets_set, assets, start_date, end_date, read_in_bonds=True, read_in_spxtr=True)
        
    # Mean-variance Optimization, Simulation and Plot Portfolios
    if(run_process_portfolios):
        for portfolio_set_name in ['standard']: # ['cov_shrinkage', 'mgarch', 'mgarch_shrinkage']:
            process_common_portfolio_set(portfolio_set_name, assets_set, subfolder_name, start_date, test_start_date, 
                                         lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint',
                                         run_rolling_window=True, run_yearly_quarterly_monthly=True, find_weights=True, find_coefs=True, 
                                         transaction_cost=0.01, run_alt_rebalance=False)
    
    # Plot efficient frontier visuals
    if(run_plot_ef_viz):
        plot_yearly_efs_over_time(assets_set, start_date.year, end_date.year)
        plot_ef_tan_port(assets_set, dt.datetime(2018, 1, 1))
    

