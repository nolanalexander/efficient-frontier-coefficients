import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def create_forecast_data_viz(assets_set):
    full_df_yearly = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_yearly.csv', index_col=0)
    full_df_quarterly = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_quarterly.csv', index_col=0, parse_dates=True)
    full_df_monthly = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_monthly.csv', index_col=0, parse_dates=True)
    full_df_rolling_1mo = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/mkt_forecast_df_rolling_1mo.csv', index_col=0, parse_dates=True)
    full_df_by_interval = {'Yearly' : full_df_yearly, 'Quarterly' : full_df_quarterly, 'Monthly' : full_df_monthly, 'Rolling_1mo' : full_df_rolling_1mo}
    
    time_intervals = ['Monthly', 'Rolling_1mo']
    predictors = ['r_MVP', 'sigma_MVP', 'u']
    predictand = 'Mkt_Next_Up'
    bins_by_time_interval = {
        'Monthly'     : 15,
        'Rolling_1mo' : 30,
    }
    
    forecast_dir = '../Portfolios/'+assets_set+'/Mkt_Forecast/'
    
    
    for time_interval in time_intervals:
        plot_dir = forecast_dir+time_interval+'/Plots/'
        cur_full_df = full_df_by_interval[time_interval].dropna()
        # Time-series Plots
        for col in predictors + [predictand]:
            plt.figure(figsize=(10,5))
            plt.plot(cur_full_df.index, cur_full_df[col])
            plt.title(col + ' Over Time')
            time_series_dir = plot_dir + 'Time_Series/'
            if not os.path.exists(time_series_dir):
                os.makedirs(time_series_dir)
            plt.savefig(time_series_dir + col +'_over_time.png')
            plt.close()
        # Histogram
        plt.figure(figsize=(10,5))
        for predictor in predictors:
            for is_up in [0, 1]:
                plt.hist(cur_full_df.loc[cur_full_df[predictand].astype(int) == is_up, predictor], 
                         label=is_up, alpha=0.8, density=True, bins=bins_by_time_interval[time_interval])
            plt.title(predictand + ' ' + predictor + ' Histograms')
            plt.legend()
            hist_dir = plot_dir + 'Histogram/'
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            plt.savefig(hist_dir + predictand +'_'+ predictor +'_histograms.png')
            plt.close()
        
        # Up/Down Univariate Scatterplots
        for predictand in [predictand]:
            for predictor in predictors:
                plt.figure(figsize=(7, 6))
                plt.scatter(cur_full_df[predictor], cur_full_df[predictand])
                plt.xlabel(predictor)
                plt.ylabel(predictand)
                scatterplot_dir = plot_dir + 'Scatterplots/'
                if not os.path.exists(scatterplot_dir):
                    os.makedirs(scatterplot_dir)
                plt.savefig(scatterplot_dir + predictand + '_vs_' + predictor +'.png')
                plt.close()
        
         # Up/Down Bivariate Scatterplots
        for predictor1 in predictors:
            for predictor2 in predictors:
                if predictor1 != predictor2:
                    plt.figure(figsize=(7, 6))
                    for is_up in [0,1]:
                        plt.scatter(cur_full_df.loc[cur_full_df[predictand] == is_up, predictor1], cur_full_df.loc[cur_full_df[predictand] == is_up, predictor2], label=is_up)
                    plt.xlabel(predictor1)
                    plt.ylabel(predictor2)
                    plt.legend()
                    scatterplot_dir = plot_dir + 'Scatterplots/'
                    if not os.path.exists(scatterplot_dir):
                        os.makedirs(scatterplot_dir)
                    plt.savefig(scatterplot_dir + predictor1 + '_vs_' + predictor2 +'.png')
                    plt.close()
        
        # ACF Plots
        for predictand in [predictand]:
            acf_dir = plot_dir + 'ACF/'
            if not os.path.exists(acf_dir):
                os.makedirs(acf_dir)
            plot_acf(cur_full_df[predictand], lags=5)
            plt.savefig(acf_dir + predictand + '_ACF_Plot.png')
            plt.close()
            plot_pacf(cur_full_df[predictand], lags=5)
            plt.savefig(acf_dir + predictand + '_PACF_Plot.png')
            plt.close()
            
