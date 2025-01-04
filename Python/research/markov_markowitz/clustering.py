import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from dtaidistance import dtw, clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

'''
Clusters by the appropriate model
'''

def cluster(df, clustering_model, num_clusters, return_model=False):
    scaled_df = pd.DataFrame(preprocessing.scale(df), columns=df.columns, index=df.index)
    n = len(scaled_df.index)
    if clustering_model == 'Kmeans':
        model = KMeans(n_clusters=num_clusters, random_state=0).fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward').fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Avg':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average').fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Complete':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='complete').fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Single':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single').fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Spectral':
        model = SpectralClustering(n_clusters=num_clusters, assign_labels='kmeans', random_state=0).fit(scaled_df)
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Corr_Avg':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average').fit(np.sqrt(2 * (1 - scaled_df.T.corr().values)))
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Corr_Complete':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete').fit(np.sqrt(2 * (1 - scaled_df.T.corr().values)))
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_Corr_Single':
        model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='single').fit(np.sqrt(2 * (1 - scaled_df.T.corr().values)))
        predictions = model.labels_
    elif clustering_model == 'Hierarchical_DTW_Ward':
        model = clustering.LinkageTree(dists_fun=dtw.distance_matrix_fast, dists_options={}, method='ward')
        linkage_matrix = model.fit(np.sqrt(2 * n * (1 - scaled_df.T.corr())).values)
        predictions = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1
        # model.plot()
#         dendrogram(linkage_matrix, labels=np.repeat('', len(scaled_df.index)))
#         import matplotlib.pyplot as plt
#         plt.gca().xaxis.set_visible(False)
#         print('')
    elif clustering_model == 'Hierarchical_DTW_Avg':
        model = clustering.LinkageTree(dists_fun=dtw.distance_matrix_fast, dists_options={}, method='average')
        linkage_matrix = model.fit(np.sqrt(2 * n * (1 - scaled_df.T.corr())).values)
        predictions = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1
    elif clustering_model == 'Hierarchical_DTW_Complete':
        model = clustering.LinkageTree(dists_fun=dtw.distance_matrix_fast, dists_options={}, method='complete')
        linkage_matrix = model.fit(np.sqrt(2 * n * (1 - scaled_df.T.corr())).values)
        predictions = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1
    elif clustering_model == 'Hierarchical_DTW_Single':
        model = clustering.LinkageTree(dists_fun=dtw.distance_matrix_fast, dists_options={}, method='single')
        linkage_matrix = model.fit(np.sqrt(2 * n * (1 - scaled_df.T.corr())).values)
        predictions = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1
    else:
        raise ValueError(clustering_model, 'is not a valid clustering model')
        
    if return_model:
        return pd.Series(predictions+1, index=scaled_df.index), model
    else:
        return pd.Series(predictions+1, index=scaled_df.index)




