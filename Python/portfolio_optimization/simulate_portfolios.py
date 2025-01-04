import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_processing.read_in_data import read_in_spxtr, read_in_bonds

'''
Given daily portfolio weights, simulates the portfolio values and plots the portfolio
'''

def get_portfolio_returns(weights_df, assets_returns_df, transaction_cost=0.01, update_freq=1, abs_dev_update=None):
    if update_freq is not None and update_freq == 1 and abs_dev_update is None:
        daily_transaction_costs =  transaction_cost * abs(weights_df-weights_df.shift(1).fillna(0)).sum(axis=1).iloc[:-1]
        port_returns_wo_fees = (weights_df.iloc[:-1].shift(1) * (np.e**assets_returns_df.loc[weights_df.index[:-1]]-1).iloc[:-1]).sum(axis=1)
        port_returns = (1 - daily_transaction_costs.shift(1).fillna(0)) * port_returns_wo_fees
    else:
        daily_transaction_costs = pd.Series(index=assets_returns_df.index[:-1], dtype=float)
        port_returns_wo_fees    = pd.Series(index=assets_returns_df.index[:-1], dtype=float)
        counter = 0
        last_weights, cur_weights = np.repeat(0, len(weights_df.columns)), None
        for i in range(0, len(weights_df.index[:-1])):
            prior_date, cur_date = weights_df.index[i-1] if i > 0 else None, weights_df.index[i]
            cur_weights_candidate = weights_df.loc[cur_date]
            abs_dev = abs(cur_weights_candidate - last_weights).sum()
            if (update_freq is not None and counter > update_freq) or (abs_dev_update is not None and abs_dev > abs_dev_update):
                last_weights = cur_weights.copy() if cur_weights is not None else last_weights
                cur_weights = cur_weights_candidate.copy()
                counter = 0
                daily_transaction_costs.loc[cur_date] = transaction_cost * abs_dev
            else:
                daily_transaction_costs.loc[cur_date] = 0
                counter += 1
            
            port_returns_wo_fees.loc[cur_date] = (weights_df.loc[prior_date] * (np.e**assets_returns_df.loc[cur_date]-1)).sum() if prior_date is not None else 0
        port_returns = (1 - daily_transaction_costs.shift(1).fillna(0)) * port_returns_wo_fees
    
    port_and_fee_rets_df = pd.DataFrame({'Port_Returns' : port_returns, 
                                         'Port_Returns_before_fees' : port_returns_wo_fees, 
                                         'Fees' : daily_transaction_costs})
    return port_and_fee_rets_df

def get_portfolio_vals(weights_df, assets_returns_df, transaction_cost=0.01, update_freq=1, abs_dev_update=None):
    port_and_fee_rets_df = get_portfolio_returns(weights_df, assets_returns_df, transaction_cost, update_freq=update_freq, abs_dev_update=abs_dev_update)
    port_and_fee_vals_df = pd.DataFrame({'Port_Vals' : (1+port_and_fee_rets_df['Port_Returns']).cumprod(), 
                                         'Port_Vals_before_fees' : (1+port_and_fee_rets_df['Port_Returns_before_fees']).cumprod(), 
                                         'Fees' : port_and_fee_rets_df['Fees']})
    return port_and_fee_vals_df

def tensor_shift_dim1(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:,:num,:] = fill_value
        result[:,num:,:] = arr[:,:-num,:]
    elif num < 0:
        result[:,num:,:] = fill_value
        result[:,:num,:] = arr[:,-num:,:]
    else:
        result[:] = arr
    return result

def get_portfolio_returns_from_tensor(weights_tensor, assets_returns_df, transaction_cost=0.01):
    daily_transaction_costs = transaction_cost * np.nan_to_num(abs(weights_tensor-tensor_shift_dim1(weights_tensor, 1)).sum(axis=2)[:,:-1], nan=0)
    port_returns_wo_fees = (weights_tensor[:,:-1,:] * assets_returns_df.shift(-1)[:-1].values.reshape(1, len(assets_returns_df.index)-1, len(assets_returns_df.columns))).sum(axis=2)
    return (1 - daily_transaction_costs) * port_returns_wo_fees

def simulate_portfolio(assets_set, subfolder_name, weights_filename, start_date, transaction_cost=0.01, update_freq=1, abs_dev_update=None, 
                       portfolio_name=None, port_name_ext='', save_folder='.'):
    weights_dir = '../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Weights/'+save_folder+'/'
    port_name_ext = port_name_ext if port_name_ext == '' or port_name_ext[0] == '_' else '_'+port_name_ext
    mark_tan_port_weights_3mo = pd.read_csv('../Portfolios/'+assets_set+'/Versions/'+subfolder_name+'/Weights/daily_weights_tan_rolling_3mo.csv', index_col=0, parse_dates=True)
    
    if weights_filename == 'eq_weights':
        weights = pd.DataFrame(1/len(mark_tan_port_weights_3mo.columns), columns=mark_tan_port_weights_3mo.columns, index=mark_tan_port_weights_3mo.index)
        assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    elif weights_filename == 'spx':
        spx_df = read_in_spxtr(assets_set, mark_tan_port_weights_3mo.index)
        weights = pd.DataFrame({'^SP500TR' : np.repeat(1,len(spx_df.index))}, index=spx_df.index)
        assets_returns_df = spx_df[['Chg']].rename(columns={'Chg' : '^SP500TR'})
    elif weights_filename == 'bonds':
        bonds_df = read_in_bonds(assets_set, mark_tan_port_weights_3mo.index)
        weights = pd.DataFrame({'FPNIX' : np.repeat(1,len(bonds_df.index))}, index=bonds_df.index)
        assets_returns_df = bonds_df[['Chg']].rename(columns={'Chg' : 'FPNIX'})
    elif weights_filename == '60/40':
        bonds_df = read_in_bonds(assets_set, mark_tan_port_weights_3mo.index)
        spx_df = read_in_spxtr(assets_set, mark_tan_port_weights_3mo.index)
        weights = pd.DataFrame({'^SP500TR' : np.repeat(0.6,len(spx_df.index)),
                                'FPNIX'    : np.repeat(0.4,len(bonds_df.index))}, index=spx_df.index)
        assets_returns_df = pd.DataFrame({'^SP500TR' : bonds_df['Chg'], 'FPNIX' : spx_df['Chg']})
    elif weights_filename == 'perfect_tan_3mo':
        weights = mark_tan_port_weights_3mo.shift(-int(252/4)).dropna()
        assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    else:
        weights = pd.read_csv(weights_dir+weights_filename, index_col=0, parse_dates=True)
        assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
        
    if(start_date is not None):
        weights = weights[weights.index >= start_date]
        assets_returns_df = assets_returns_df[assets_returns_df.index >= start_date]
        
    full_port_vals_df = get_portfolio_vals(weights, assets_returns_df, transaction_cost=transaction_cost, update_freq=update_freq, abs_dev_update=abs_dev_update)
    if portfolio_name is not None:
        full_port_vals_df.insert(0, 'Portfolio', portfolio_name)
    full_port_vals_df.index = full_port_vals_df.index.rename('Date')
    if portfolio_name is not None:
        full_port_vals_df = full_port_vals_df.reset_index()
    return full_port_vals_df

def plot_portfolios(port_vals_df, label_by_port, title, plot_dir, plot_filename, scale='log'):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plt.figure(figsize=(10, 5))
    for port_name in label_by_port:
        if scale == 'log':
            port_vals = np.log(port_vals_df[port_name])
            ylabel = 'Log Portfolio Value'
        elif scale == 'raw':
            port_vals = port_vals_df[port_name]
            ylabel = 'Portfolio Value'
        elif scale == 'percent':
            port_vals = port_vals_df[port_name] * 100
            ylabel = '% Portfolio Value'
        else:
            raise ValueError(f'Invalid scale: {scale}')
        plt.plot(port_vals_df.index, port_vals, label=label_by_port[port_name])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.savefig(plot_dir+plot_filename)
    plt.close()
    
