Implementation of three journal/conference papers on Efficient Frontier Coefficients.

## Overview
This project contains the code for my research in Efficient Frontier Coefficients throughout my undergrad and Master's. This research was collected into three papers: two journal papers and one conference paper. The code for the implementation of these three papers all work, but the code could use some refactoring.

The project also includes code for various avenues of research that were not fully explored including feature selection, options, and reinforcement learning. The code for these unfinished research areas may not be fully functional.

## Publications
1. [Asset allocation using a Markov process of clustered efficient frontier coefficients states](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=vm_4mhoAAAAJ&authuser=1&citation_for_view=vm_4mhoAAAAJ:2osOgNQ5qMEC)
2. [Forecasting Tangency Portfolios and Investing in the Minimum Euclidean Distance Portfolio to Maximize Out-of-Sample Sharpe Ratios](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=vm_4mhoAAAAJ&authuser=1&citation_for_view=vm_4mhoAAAAJ:9yKSN-GCB0IC)
3. [Using Machine Learning to Forecast Market Direction with Efficient Frontier Coefficients](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=vm_4mhoAAAAJ&authuser=1&citation_for_view=vm_4mhoAAAAJ:IjCSPb-OGe4C)

## File Structure and Descriptions

At the highest level there are three folders: "CSV_Data", "Portfolios", and "Python". CSV_Data contains collected CSV files. "Portfolios" is where all results are stored for each set of assets. "Python" is python code organized as follows:

- **main_\***: all files beginning with "main_" are the pipeline runner files. They contain booleans for each which part of the pipeline to run.
- **data_processing**: collecting and processing data
    - **read_in_data.py**: reads in Yahoo finance data
    - **data_cleaning.py**: handles missing days and multi-index transformations
    - **preprocess_data.py**: reformats and cleans data
    - **ts_utils.py**: time-series utility functions

- **forecast_backtest**: backtest forecasts
    - **ForecastModel.py**: defines models with methods to fit, tune, and forecast
    - **walk_forward_forecast.py**: forecast with variable train/tune updates
    - **backtest_forecasts.py**: backtest continuous/binary strategies from forecasts
    - **forecast_metrics.py**: forecasting metrics

- **portfolio_optimization**: mean-variance portfolio optimization
    - **assets_selection.py**: asset selection methods based on correlation and PCA Biplot
    - **markowitz.py**: mean-variance optimization, MVP, tangency portfolio, EF coefs
    - **mean_estimation.py**: mean estimator funcs to be passed in to mean-variance optimizer
    - **cov_estimation.py**: cov estimator funcs to be passed in to mean-variance optimizer
    - **common_portfolios.py**: common mean-variance portfolios with different lookbacks
    - **portfolio_sets_inventory.py**: portfolio parameter estimators sets
    - **black_litterman.py**: Black-Litterman model as mean and cov estimator
    - **portfolio_utils.py**: weights scaling, portfolio metrics, alpha regression

- **research**: published research code
    - **markov_markowitz**: code for paper 1
        - **clustering.py**: unsupervised learning methods
        - **markov_model.py**: Markov process
        - **markov_markowitz_model.py**: state-transition portfolio model
        - **prep_markov_markowitz_sim.py**: preparation for backtest
        - **simulate_markov_markowitz.py**: backtest of model
        - **visualizations**
            - **graph_markov_model_full.py**: graph state transition
            - **plot_clustering_full.py**: plot clustering
    - **markowitz_ext**: code for paper 2
        - **setup_markowitz_ext_forecast.py**: calculate EF coefs for regression
        - **markowitz_ext_forecast.py**: forecast tangency portfolios
        - **markowitz_ext_weights_extrac.py**: convert forecast to weights 
        - **simulate_mark_ext_portfolios.py**: backtest model
        - **visualizations**
            - **exploratory_analysis.py**: PCA, scatterplots, etc.
            - **plot_mark_ext_ef_vis.py**: plot forecasted tangency portfolio 
    - **mkt_forecast_markowitz**: code for paper 3
        - **create_mkt_forecast_data.py**: calculate EF coefs for forecast
        - **forecast_mkt.py**: forecast with ML on EF coefs data
        - **conditional_expectation_portfolio.py**: convert forecast to conditional expectation with inverse Mills ratio
        - **simulate_forecast_mkt_portfolio.py**: backtest model
        - **visualizations**
            - **backtest_forecast_spx_pos.py**: plot forecasts
            - **forecast_data_viz.py**: plot EF coefs
            
- **ts_features**: time-series features. This was going to be for creating large feature sets to test alongside EF coefs.
    - **feature_engr.py**: common technical indicators
    - **feature_selection.py**: VIF, and information theoretic feature selection (MRMR)      
    
- **options**: options backtesting. This was going to be for using EF coefs to time volatility selling strategies.
    - **black_scholes_greeks.py**: Black-Scholes, implied vol, greeks, and shadow greeks
    - **clean_options_data.py**: not finished
    - options_strategy_backtester.py: not finished
            
- **reinforcement_learning_trading**: a stripped-down and simplified version of FinRL. This was going to be for testing with reinforecment learning with EF coefs data.
    - **AssetTradingEnv.py**: environment that realizes trades and updates rewards
    - **config.py**: DRL model settings
    - **DRLAgent.py**: agent than can go long, short, or not trade 1 unit of each asset
    - **rl_backtest.py**: runs backtest, plots training, and computes metrics
    - **run_rl_trading.py**: main runner
