import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import os

from data_processing.read_in_data import read_in_rf


def plot_yearly_efs_over_time(asset_set, start_year, end_year, xlim=None, ylim=None, num_points=10000):
    ef_coefs_yearly = pd.read_csv('../Portfolios/'+asset_set+'/EF_Coefs/ef_coefs_yearly.csv', index_col=0)
    
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["royalblue", "firebrick"])
    cnorm = mcol.Normalize(vmin=start_year, vmax=end_year)
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    
    fig = plt.figure()
    for year in ef_coefs_yearly.index:
        r_MVP, sigma_MVP, u = ef_coefs_yearly.loc[year, ['r_MVP', 'sigma_MVP', 'u']]
        rs = np.linspace(r_MVP, r_MVP + 0.005, num_points)
        sds = np.sqrt((u**-1*(rs - r_MVP))**2 + sigma_MVP**2)
        # rs = rs[sds < (sigma_MVP + 0.008)]
        # sds = sds[sds < (sigma_MVP + 0.008)]
        plt.plot(sds, rs, color=cpick.to_rgba(year))
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Return')
    if xlim is None:
        if asset_set == 'GVMC':
            xlim=[0.002, 0.017]
            plt.xlim(xlim)
        elif asset_set == 'Sectors':
            xlim=[0.003, 0.017]
            plt.xlim(xlim)
        elif asset_set == 'Sectors_and_Bonds':
            xlim=[0.0004, 0.0025]
            plt.xlim(xlim)
    if ylim is None:
        if asset_set == 'GVMC':
            ylim=[-0.0007, 0.0025]
            plt.ylim(ylim)
        elif asset_set == 'Sectors':
            ylim=[-0.0008, 0.0025]
            plt.ylim(ylim)
        elif asset_set == 'Sectors_and_Bonds':
            ylim=[-0.0002, 0.0008]
            plt.ylim(ylim)
    plt.title(f"Efficient Frontiers {start_year}-{end_year}")
    plt.colorbar(cpick, label='Year')
    plot_dir = '../Portfolios/'+asset_set+'/EF_Plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + f'efs_yearly_{start_year}to{end_year}.png', dpi=fig.dpi, bbox_inches='tight')
    plt.close()

def plot_ef_tan_port(asset_set, ef_date, xlim=None, ylim=None, num_points=10000):
    ef_coefs_monthly = pd.read_csv('../Portfolios/'+asset_set+'/EF_Coefs/ef_coefs_monthly.csv', index_col=0, parse_dates=True)
    rfs = read_in_rf(ef_coefs_monthly.index)
    
    A, B, C = ef_coefs_monthly.loc[ef_date, ['A', 'B', 'C']]
    r_MVP, sigma_MVP, u = ef_coefs_monthly.loc[ef_date, ['r_MVP', 'sigma_MVP', 'u']]
    r_f = rfs.loc[ef_date]
    r_TP = (r_MVP**2 + u**2 * sigma_MVP**2 - r_MVP * r_f) / (r_MVP - r_f)
    
    fig = plt.figure()
    rs = np.linspace(r_MVP - r_TP - 0.005, r_MVP + r_TP + 0.005, num_points)
    sds = np.sqrt((u**-1*(rs - r_MVP))**2 + sigma_MVP**2)
    sds = np.sqrt((A*rs**2 - 2 * B * rs + C) / (A*C-B**2))
    plt.plot(sds, rs, label='Efficient Frontier', zorder=1, alpha=0.8)
    
    rs = np.linspace(r_f, r_MVP + r_TP + 0.005, num_points)
    beta = (r_TP - r_MVP) / (u**2 * (np.sqrt((u**-1 * (r_TP-r_MVP))**2 + sigma_MVP**2)))
    cml_sd = beta * (rs - r_f) + 0
    plt.plot(cml_sd, rs, label='Capital Market Line', zorder=2, alpha=0.8)
    
    sigma_TP = np.sqrt((u**-1*(r_TP - r_MVP))**2 + sigma_MVP**2)
    plt.scatter(sigma_TP, r_TP, label = 'Tangency Portfolio', linewidths=0.01, c='black', zorder=3)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Return')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title('Efficient Frontier Tangency Portfolio')
    plt.legend(loc = 'upper left')
    plot_dir = '../Portfolios/'+asset_set+'/EF_Plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + 'ef_tan_port.png', dpi=fig.dpi, bbox_inches='tight')
    plt.close()
