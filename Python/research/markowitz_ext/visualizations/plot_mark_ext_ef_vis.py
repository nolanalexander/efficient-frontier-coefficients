import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
from scipy.optimize import minimize


def plot_ef_min_dist_port(asset_set, subfolder_name, ef_date):
    ef_coefs_3mo_rolling = pd.read_csv('../Portfolios/'+asset_set+'/EF_Coefs/daily_ef_coefs_tan_rolling_3mo.csv', index_col=0, parse_dates=True)
    ef_coefs_forecast = pd.read_csv('../Portfolios/'+asset_set+'/Versions/'+subfolder_name+'/Forecasted_EF_Coefs/forecasted_EF_coefs_rolling_3mo.csv', index_col=0, parse_dates=True)
    rfs = pd.read_csv('../Mkt_Indicators_Data/F-F_Research_Data_Factors_daily.csv', 
                            skiprows=4, skipfooter=2, engine='python', index_col=0)['RF']/100
    rfs.index = pd.to_datetime(rfs.index, format='%Y%m%d')
    
    # Plot EF
    A, B, C = ef_coefs_3mo_rolling.loc[ef_date, ['A', 'B', 'C']]
    r_MVP, sigma_MVP, u = ef_coefs_3mo_rolling.loc[ef_date, ['r_MVP', 'sigma_MVP', 'u']]
    fig = plt.figure(figsize=(11,6.5))
    rs = np.linspace(r_MVP - 0.01, r_MVP + 0.01, 10000)
    sds = np.sqrt((u**-1*(rs - r_MVP))**2 + sigma_MVP**2)
    sds = np.sqrt((A*rs**2 - 2 * B * rs + C) / (A*C-B**2))
    plt.plot(sds, rs, label='Historical Efficient Frontier', zorder=1, alpha=0.8)
    
    def interp_coefs_to_ABC(interp_coefs):
        return pd.Series([1/interp_coefs[1]**2, interp_coefs[0]/interp_coefs[1]**2, 
                          interp_coefs[2]**2 + interp_coefs[0]**2/interp_coefs[1]**2], 
                         index=['A', 'B', 'C'])
    
    # Plot Forecasted EF
    r_MVP_hat, sigma_MVP_hat, u_hat = ef_coefs_forecast.loc[ef_date, ['r_MVP_fwd3mo_MA', 'sigma_MVP_fwd3mo_MA', 'u_fwd3mo_MA']]
    A_hat, B_hat, C_hat = interp_coefs_to_ABC([r_MVP_hat, sigma_MVP_hat, u_hat])
    rs = np.linspace(r_MVP_hat - 0.01, r_MVP_hat + 0.01, 10000)
    sds = np.sqrt((u_hat**-1*(rs - r_MVP))**2 + sigma_MVP_hat**2)
    sds = np.sqrt((A_hat*rs**2 - 2 * B_hat * rs + C_hat) / (A_hat*C_hat-B_hat**2))
    plt.plot(sds, rs, label='Forecasted Efficient Frontier', zorder=2, alpha=0.8)
    
    # Forecasted Tangency Portfolio
    r_f = rfs.loc[ef_date]
    rs = np.linspace(r_f, r_MVP_hat + 0.01, 10000)
    r_TP = (r_MVP_hat**2 + u_hat**2 * sigma_MVP_hat**2 - r_MVP_hat * r_f) / (r_MVP_hat - r_f)
    # print('r_TP', r_TP)
    beta = (r_TP - r_MVP_hat) / (u_hat**2 * (np.sqrt((u_hat**-1 * (r_TP-r_MVP_hat))**2 + sigma_MVP_hat**2)))
    cml_sd = beta * (rs - r_f) + 0
    plt.plot(cml_sd, rs, label='Forecasted Capital Market Line', zorder=3, alpha=0.8)
    
    sigma_TP = np.sqrt((u_hat**-1*(r_TP - r_MVP_hat))**2 + sigma_MVP_hat**2)
    plt.scatter(sigma_TP, r_TP, label = 'Forecasted Tangency Portfolio', linewidths=0.01, c='C1', zorder=4)
    
    # Min Dist Portfolio
    forecasted_tan_port_ret = (C_hat - B_hat * r_f)/(B_hat - A_hat * r_f)
    forecasted_tan_port_sd = np.sqrt((A_hat * forecasted_tan_port_ret**2 - 2 * B_hat * forecasted_tan_port_ret + C_hat)/(A_hat * C_hat - B_hat**2))
    
    def get_ef_sigma(r):
        return np.sqrt((A * r**2 - 2 * B * r + C) /(A * C - B**2))
    
    def get_euclidean_dist(x1, x2):
        return np.sqrt(sum((x1-x2)**2))
    
    def distance_to_tan_port(r):
        return get_euclidean_dist(np.array([r[0], get_ef_sigma(r[0])]), np.array([forecasted_tan_port_ret, forecasted_tan_port_sd]))
    
    min_dist_port_ret = r_MVP_hat
    bounds = ((r_MVP_hat, 1.0),) # only on upper half of efficient frontier
    optimization = minimize(distance_to_tan_port, min_dist_port_ret, method='SLSQP' ,
                            bounds=bounds, options = {'disp':False, 'ftol': 1e-12, 'maxiter': 1e5} ,
                            constraints=[])# , callback=print)
    if(not optimization.success):
        print(optimization)
    min_dist_port_ret = optimization.x[0]
    min_dist_port_sd = get_ef_sigma(min_dist_port_ret)
    plt.scatter(min_dist_port_sd, min_dist_port_ret, label = 'Min Euclidean Distance Portfolio', linewidths=0.01, c='C0', zorder=5)
    plt.plot([min_dist_port_sd, forecasted_tan_port_sd], [min_dist_port_ret, forecasted_tan_port_ret], label='Distance to Forecasted Tangency Portfolio', zorder=3)
    
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Return')
    plt.xlim([0, 0.003])
    plt.ylim([-0.0005, 0.0015])
    plt.title('Efficient Frontier Min Distance Portfolio to the Tangency Portfolio')
    plt.legend(loc = 'upper left')
    plot_dir = '../Portfolios/'+asset_set+'/EF_Plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + 'ef_min_dist_port.png', dpi=fig.dpi, bbox_inches='tight')
    plt.close()
