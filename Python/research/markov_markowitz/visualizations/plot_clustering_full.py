import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from scipy.sparse import csgraph
import scipy as sp
import time
import os
from matplotlib.ticker import MaxNLocator
from dtaidistance import dtw, clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from portfolio_optimization.portfolio_utils import get_assets_set_abr
from data_processing.ts_utils import get_quarters
from data_processing.read_in_data import read_in_rf
from research.markov_markowitz.clustering import cluster
from research.markov_markowitz.prep_markov_markowitz_sim import calc_tan_port_weights_in_cluster


def plot_clustering_models_full(assets_set, cluster_features, version_name, clustering_models, num_clusters_list, time_intervals, test_start_date,
                                weights_sum_list=[0, 1],
                                lower_bound=None, upper_bound=None, max_leverage=1.5, max_leverage_method='constraint'):
    start_time = time.time()
    markov_dir = '../Portfolios/'+assets_set+'/Markov/Full_Cluster/'
    assets_set_abr = get_assets_set_abr(assets_set)
    ef_coefs_df_yearly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_yearly.csv', index_col=0)
    ef_coefs_df_quarterly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_quarterly.csv', index_col=0, parse_dates=True)
    ef_coefs_df_monthly = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_monthly.csv', index_col=0, parse_dates=True)
    ef_coefs_df_rolling_1mo = pd.read_csv('../Portfolios/'+assets_set+'/EF_Coefs/ef_coefs_rolling_1mo.csv', index_col=0, parse_dates=True)
    assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
    rfs = read_in_rf(assets_returns_df.index)
    ef_coefs_df_by_interval = {'Yearly' : ef_coefs_df_yearly, 'Quarterly' : ef_coefs_df_quarterly, 'Monthly' : ef_coefs_df_monthly, 'Rolling_1mo' : ef_coefs_df_rolling_1mo}


    for time_interval in time_intervals:
        metric_df = pd.DataFrame(columns=['Clustering_Model', 'Num_Clusters', 'Max_Cluster_Prop', 'Silhouette_Score'])
        for clustering_model in clustering_models:
            cur_ef_coefs_df = ef_coefs_df_by_interval[time_interval][cluster_features].dropna().copy()
            scaled_df = pd.DataFrame(preprocessing.scale(cur_ef_coefs_df), columns=cur_ef_coefs_df.columns, index=cur_ef_coefs_df.index)
            scaled_train_df = scaled_df[scaled_df.index < test_start_date]
            
            tuning_plot_dir = markov_dir+'/'+time_interval+'/Tuning_Graphs/'+version_name+'/'
            if not os.path.exists(tuning_plot_dir):
                os.makedirs(tuning_plot_dir)
            if(clustering_model == 'Kmeans'):
                # Elbow Graph
                kmeans = [KMeans(n_clusters=i) for i in range(1, 10)]
                score = [kmeans[i].fit(scaled_train_df).score(scaled_train_df) for i in range(len(kmeans))]
                plt.figure(figsize=(10,7))
                plt.plot(range(1, 10), score)
                plt.title('Efficient Frontier Coefficients Clusters Elbow Curve')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Score')
                plt.savefig(tuning_plot_dir+time_interval.lower()+'_ef_coefs'+'_elbow_curve.png', bbox_inches='tight')
                plt.close()
            elif(clustering_model == 'Hierarchical'):
                # Hierarchy Visualization with Dendrogram
                plt.figure(figsize=(10,7))
                dend = shc.dendrogram(shc.linkage(scaled_train_df, method='ward'), no_labels=True) #(len(scaled_df.index) > 40))
                # plt.axhline(y=4, color='r', linestyle='--')
                plt.title("Efficient Frontier Coefficients Clustering Dendrogram")
                plt.ylabel('Distance')
                plt.xlabel(time_interval[:-2])
                plt.savefig(tuning_plot_dir + time_interval.lower()+'_ef_coefs'+'_dendrogram.png', bbox_inches='tight')
                plt.close()
            elif(clustering_model == 'Spectral'):
                # PCA 
                spectral = SpectralClustering().fit(scaled_train_df)
                aff_mat = spectral.affinity_matrix_
                lap_aff_mat = csgraph.laplacian(aff_mat, normed=True)
                pca = PCA()
                principalComponents = pca.fit_transform(lap_aff_mat)
                exp_var_pca = pca.explained_variance_ratio_[:10]
                cum_sum_eigenvalues = np.cumsum(exp_var_pca)
                plt.figure(figsize=(10,7))
                plt.plot(range(1,10+1), cum_sum_eigenvalues)
                # plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual')
                # plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative')
                plt.title('Efficient Frontier Coefficients Laplacian Affinity Matrix PCA')
                plt.ylabel('Proportion of Explained Variance')
                plt.xlabel('Laplacian Affinity Matrix Principal Component')
                # plt.tight_layout()
                plt.savefig(tuning_plot_dir + time_interval.lower()+'_ef_coefs'+'_pca_explained_variance.png', bbox_inches='tight')
                plt.close()
            elif clustering_model == 'Hierarchical_DTW_Ward':
                n = len(scaled_df.index)
                model = clustering.LinkageTree(dists_fun=dtw.distance_matrix_fast, dists_options={}, method='ward')
                linkage_matrix = model.fit(np.sqrt(2 * n * (1 - scaled_train_df.T.corr())).values)
                plt.figure(figsize=(10,7))
                dendrogram(linkage_matrix, labels=np.repeat('', len(scaled_train_df.index)))
                plt.title(assets_set_abr + ' Dendrogram')
                plt.ylabel('Distance')
                plt.gca().xaxis.set_visible(False)
                plt.savefig(tuning_plot_dir + time_interval.lower()+'_ef_coefs'+'dtw_ward_dendrogram.png', bbox_inches='tight', dpi=200)
                plt.close()
            else:
                pass
                # raise ValueError(clustering_model, 'is not a valid clustering model')
            
            for num_clusters in num_clusters_list:
                predictions = cluster(scaled_df, clustering_model, num_clusters)
                train_predictions = cluster(scaled_train_df, clustering_model, num_clusters)
                
                # Measure model performance using silhouette score [-1,1], where closer to 1, the clusters are more dense
                max_cluster_prop = round(sp.stats.mode(train_predictions)[1][0]/len(train_predictions), 2)
                metric_df.loc[len(metric_df.index)] = [clustering_model, num_clusters, max_cluster_prop, silhouette_score(scaled_train_df, train_predictions)]
                
                plot_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'Cluster_Graphs/'+version_name+'/'
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                
                # Scatterplot matrix 
                num_features = len(scaled_df.columns)
                fig, ax = plt.subplots(num_features, num_features, figsize=(12,12))
                for i in range(num_features):
                    for j in range(num_features):
                        colors = {1:'tab:blue', 2:'tab:orange', 3:'tab:green', 4:'tab:red', 5:'tab:purple', 6:'tab:brown', 7:'tab:pink', 8:'tab:grey'}
                        ax[i, j].scatter(scaled_df[scaled_df.columns[i]], scaled_df[scaled_df.columns[j]], c=predictions.map(colors))
                        ax[i, j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                        ax[i, j].axis('equal')
                for i in range(num_features):
                    ax[i, 0].set_ylabel(scaled_df.columns[i], fontsize=20)
                    ax[num_features-1, i].set_xlabel(scaled_df.columns[i], fontsize=20)
                plt.suptitle(assets_set_abr + ' Clustering of Efficient Frontier Coefficients', fontsize=25)
                fig.savefig(plot_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_scatter_matrix.png', bbox_inches='tight', dpi=200)
                plt.close()
                
                # 3-space Scatterplot
                fig = plt.figure(figsize=(10, 9))
                ax = Axes3D(fig, elev=30, azim=130)
                #ax = fig.add_subplot(111, projection='3d')
                ax.scatter(scaled_df[cluster_features[0]], scaled_df[cluster_features[1]], scaled_df[cluster_features[2]], c=predictions.astype(np.float), edgecolor='k')
                # if(clustering_model == 'Kmeans'):
                #     ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
                ax.w_xaxis.set_ticklabels([])
                ax.w_yaxis.set_ticklabels([])
                ax.w_zaxis.set_ticklabels([])
                ax.set_xlabel(cluster_features[0])
                ax.set_ylabel(cluster_features[1])
                ax.set_zlabel(cluster_features[2])
                ax.set_title(assets_set_abr + ' Clustering of Efficient Frontier Coefficients with ' + str(num_clusters) + ' Clusters')
                fig.savefig(plot_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_3d_scatterplot.png', bbox_inches='tight')
                plt.close()
                
                # States Over Time Plot
                fig = plt.figure(figsize=(10, 4))
                if time_interval == 'Monthly':
                    recessions = pd.read_csv('../CSV_Data/USREC.csv', index_col=0, parse_dates=True)
                    recessions = recessions.loc[predictions.index, 'USREC']
                    plt.scatter(predictions.index[recessions == 0], predictions[recessions == 0], label='Not Recession')
                    plt.scatter(predictions.index[recessions == 1], predictions[recessions == 1], c='firebrick', label='Recession')
                    plt.legend()
                elif time_interval == 'Quarterly':
                    recessions = pd.read_csv('../CSV_Data/USRECQ.csv', index_col=0, parse_dates=True)
                    recessions = recessions.loc[predictions.index, 'USRECQ']
                    plt.scatter(predictions.index[recessions == 0], predictions[recessions == 0], label='Not Recession')
                    plt.scatter(predictions.index[recessions == 1], predictions[recessions == 1], c='firebrick', label='Recession')
                else:
                    plt.scatter(predictions.index, predictions)
                fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                plt.title(assets_set_abr + ' Clustered States Over Time')
                plt.ylabel('State')
                fig.savefig(plot_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_states_over_time.png', bbox_inches='tight', dpi=200)
                plt.close()
                
                # Save Clustering Results
                results_df = pd.DataFrame({'State': predictions}, index=scaled_df.index)
                results_df = pd.concat([results_df, scaled_df], axis=1)
                results_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'Cluster_Results/'+version_name+'/'
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                results_df.to_csv(results_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_clustering_results.csv')
                
                # Calculate Market Summary by Cluster
                assets_returns_df = pd.read_csv('../Portfolios/'+assets_set+'/Assets_Data/assets_returns_data.csv', index_col=0, parse_dates=True)
                ew_ret_df = pd.DataFrame({'Chg' : assets_returns_df.sum(1)/len(assets_returns_df.columns)})
                
                ew_ret_df['Year'] = ew_ret_df.index.year
                ew_ret_df['Quarter'] = get_quarters(ew_ret_df.index)
                ew_ret_df['Month'] = [dt.datetime(date.year, date.month, 1) for date in ew_ret_df.index]
                if time_interval[:7] != 'Rolling':
                    ew_ret_mkt_cluster_df = ew_ret_df.merge(results_df, left_on=time_interval[:-2], right_index=True)
                else:
                    ew_ret_mkt_cluster_df = ew_ret_df.merge(results_df, left_index=True, right_index=True)
                ew_ret_mkt_cluster_df = ew_ret_mkt_cluster_df[['State', 'Chg']]
                
                mkt_cluster_df = pd.DataFrame(index=range(1, len(results_df['State'].unique())+1))
                mkt_cluster_df['Mkt_Ret'] = ew_ret_mkt_cluster_df.groupby('State')['Chg'].mean()
                mkt_cluster_df['Mkt_SD'] = ew_ret_mkt_cluster_df.groupby('State')['Chg'].std()
                # mkt_cluster_df['Mkt_SD'] = mkt_cluster_df['Mkt_SD'] * np.sqrt(252)
                mkt_cluster_df['Mkt_SD'] = np.sqrt((mkt_cluster_df['Mkt_SD']**2 + (1+mkt_cluster_df['Mkt_Ret'])**2)**252 - ((1+mkt_cluster_df['Mkt_Ret'])**2)**252)
                mkt_cluster_df['Mkt_Ret'] = (1 + mkt_cluster_df['Mkt_Ret']) ** 252 - 1
                # mkt_cluster_df['Mkt_Sharpe'] = mkt_cluster_df['Mkt_Ret']/mkt_cluster_df['Mkt_SD'] * np.sqrt(252)
                
                cluster_mkt_summary_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'Cluster_Mkt_Summary/'+version_name+'/'
                if not os.path.exists(cluster_mkt_summary_dir):
                    os.makedirs(cluster_mkt_summary_dir)
                mkt_cluster_df.to_csv(results_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_mkt_summary.csv')
                
                # Create Market Summary Barchart
                mkt_cluster_df['Pos'] = mkt_cluster_df['Mkt_Ret'] > 0
                ax = plt.figure().gca()
                plt.bar(mkt_cluster_df.index.astype(int), mkt_cluster_df['Mkt_Ret'] * 100, color=mkt_cluster_df['Pos'].map({True: 'g', False: 'r'}),
                        yerr=mkt_cluster_df['Mkt_SD'] * 100, align='center', ecolor='black')
                plt.title('Equal-Weight Portfolio Annual Returns in '+ assets_set_abr + ' Clustered States')
                plt.xlabel('State')
                plt.ylabel('Equal-Weight Portfolio Annual Return (%)')
                ax.xaxis.get_major_locator().set_params(integer=True)
                plt.savefig(cluster_mkt_summary_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_mkt_summary_barplot.png', dpi=200)
                plt.close()
                
                # Calculate Final State Weights
                assets_returns_df['Year'] = assets_returns_df.index.year
                assets_returns_df['Quarter'] = get_quarters(assets_returns_df.index)
                assets_returns_df['Month'] = [dt.datetime(date.year, date.month, 1) for date in assets_returns_df.index]
                if time_interval[:7] != 'Rolling':
                    clustered_assets_returns_df = assets_returns_df.merge(results_df[['State']], left_on=time_interval[:-2], right_index=True).copy()
                else:
                    clustered_assets_returns_df = assets_returns_df.merge(results_df[['State']], left_index=True, right_index=True).copy()
                clustered_assets_returns_df = clustered_assets_returns_df.drop(columns=['Year', 'Quarter', 'Month'])
                clustered_assets_returns_df.columns = ['^SPX' if col == '^SP500TR' else col for col in clustered_assets_returns_df.columns]
                full_state_weights_df = clustered_assets_returns_df.groupby('State').apply(calc_tan_port_weights_in_cluster, rf=rfs.iloc[-1],
                                                                                            weights_sum_list=weights_sum_list, lower_bound=lower_bound, upper_bound=upper_bound,
                                                                                            max_leverage=max_leverage, max_leverage_method=max_leverage_method)
                state_weights_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'State_Weights/'+version_name+'/'
                if not os.path.exists(state_weights_dir):
                    os.makedirs(state_weights_dir)
                full_state_weights_df.to_csv(state_weights_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_state_weights_full.csv')
#                 full_state_weights_df.loc[1] = pd.Series(full_state_weights_df.loc[1].values.reshape(2,-1).ravel(order='F'), full_state_weights_df.columns)
                
                # Plot State Weights Heatmap
                cmap = plt.get_cmap('coolwarm')
                cmaplist = [cmap(i) for i in range(cmap.N)]
                cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
                bounds = np.arange(np.min(full_state_weights_df.dropna().values),np.max(full_state_weights_df.dropna().values),0.2)
                idx=np.searchsorted(bounds,0)
                bounds=np.insert(bounds,idx,0)
                norm = BoundaryNorm(bounds, cmap.N)
                fig = plt.figure()
                ax = sns.heatmap(full_state_weights_df, linewidth = 0.5, cmap=cmap, norm=norm, center=0)
                plt.title(assets_set_abr + ' State Weights Heatmap')
                plt.xlabel('Portfolio Weights')
                plt.ylabel('State')
                markov_dir = '../Portfolios/'+assets_set+'/Markov/Full_Cluster/'
                state_weights_dir = markov_dir+time_interval+'/'+str(num_clusters)+'_Clusters/'+'State_Weights/'
                plt.savefig(state_weights_dir + clustering_model.lower() + '_' + str(num_clusters) + '_clusters_' + time_interval.lower() + '_state_weights_full_heatmap.png', dpi=200)
                plt.close()
                
            # for clustering_model in clustering_models:
            #     plt.figure(figsize=(10,5))
            #     cur_metric_df = metric_df[metric_df['Clustering_Model']== clustering_model]
            #     plt.plot(cur_metric_df['Num_Clusters'], cur_metric_df['Silhouette_Score'])
            #     plt.title(assets_set_abr + ' ' + clustering_model.replace("_", " ")[:16] + ' Silhouette Scores')
            #     plt.ylabel('Silhouette Score')
            #     plt.xlabel('Number of Clusters')
            #     plt.savefig(tuning_plot_dir + clustering_model.lower() + '_' + time_interval.lower() + '_silhouette_scores.png')
            #     plt.close()
        
                
        metrics_dir = markov_dir+'/'+time_interval+'/Metrics/'+version_name+'/'
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        metric_df['Max_Silhouette_Score'] = (metric_df['Silhouette_Score'] == max(metric_df['Silhouette_Score'])).astype(int)
        metric_df.to_csv(metrics_dir+time_interval.lower()+'_silhouette_scores.csv', index=False)
    print('Full Data Clustering Run Time:', round((time.time() - start_time)/60), 'mins')




