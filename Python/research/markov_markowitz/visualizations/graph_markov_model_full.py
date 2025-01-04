import numpy as np
import pandas as pd
import os
import networkx as nx
import pydot
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns

from research.markov_markowitz.markov_model import calc_transition_matrix, calc_steady_state_prob
from portfolio_optimization.portfolio_utils import get_assets_set_abr

def get_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            if(round(Q.loc[idx,col], 2) != 0):
                edges[(idx,col)] = round(Q.loc[idx,col], 2)
    return edges

def graph_markov_model_full(assets_set, cluster_features, version_name, clustering_models, num_clusters_list, time_intervals):
    markov_dir = '../Portfolios/'+assets_set+'/Markov/Full_Cluster/'
    assets_set_abr = get_assets_set_abr(assets_set)
    
    # To handle pydot errno error message
    import errno
    setattr(os, 'errno', errno)
    
    for time_interval in time_intervals:
        for clustering_model in clustering_models:
            for num_clusters in num_clusters_list:
                
                # Calculate Transition Matrix
                results_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'Cluster_Results/'+version_name+'/'
                states_over_time = pd.read_csv(results_dir+clustering_model.lower()+'_'+str(num_clusters)+'_clusters_'+time_interval.lower()+'_clustering_results.csv')['State']
                state_trans_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'State_Transitions/'+version_name+'/'
                if not os.path.exists(state_trans_dir):
                    os.makedirs(state_trans_dir)
                transition_matrix = calc_transition_matrix(states_over_time)
                transition_matrix.to_csv(state_trans_dir+clustering_model.lower()+'_'+str(num_clusters)+'_clusters_'+time_interval.lower()+'_transition_matrix.csv')
                
                # Calculate steady state probs
                steady_state_prob = calc_steady_state_prob(transition_matrix)
                steady_state_prob_df = pd.DataFrame({'Steady_State_Prob' : steady_state_prob}, index=range(1, len(steady_state_prob)+1))
                steady_state_prob_df.to_csv(state_trans_dir+clustering_model.lower()+'_'+str(num_clusters)+'_clusters_'+time_interval.lower()+'_steady_state_probs.csv')
                
                # Graph Markov State Transition Diagram
                edge_wts = get_edges(transition_matrix)
                G = nx.MultiDiGraph()
                G.add_nodes_from(range(1,num_clusters))
                for k, v in edge_wts.items():
                    tmp_origin, tmp_destination = k[0], k[1]
                    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
                graph = nx.drawing.nx_pydot.to_pydot(G)
                graph.write_png(state_trans_dir+clustering_model.lower()+'_'+str(num_clusters)+'_clusters_'+time_interval.lower()+'_markov_diagram.png')
                
                # Graph State Transition Heatmap
                cmap = plt.get_cmap('coolwarm')
                cmaplist = [cmap(i) for i in range(cmap.N)]
                cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
                bounds = np.arange(0, 1.1, 0.2)
                idx = np.searchsorted(bounds,0)
                bounds = np.insert(bounds,idx,0)
                norm = BoundaryNorm(bounds, cmap.N)
                fig = plt.figure()
                ax = sns.heatmap(transition_matrix, linewidth = 0.5, cmap=cmap, vmax=1, vmin=0, center=0)
                plt.title(assets_set_abr + ' Transition Matrix')
                plt.xlabel('State')
                plt.ylabel('State')
                plt.savefig(state_trans_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_transition_matrix_heatmap.png', dpi=200)
                plt.close()
