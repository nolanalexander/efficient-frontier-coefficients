import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
# pip install adjustText

from research.markowitz_ext.markowitz_ext_forecast import get_regr_data


def exploratory_analysis(asset_set, lookback_method):
    start_time = time.time()
    
    coefs_regr_df, predictors, predictands = get_regr_data(asset_set, lookback_method)
    coefs_regr_df_copy = coefs_regr_df.copy()
    
    expl_dir = '../Portfolios/'+asset_set+'/Markowitz_Ext/Exploratory_Analysis/'+lookback_method+'/'
    if not os.path.exists(expl_dir):
        os.makedirs(expl_dir)
    
    corr_mat = round(coefs_regr_df_copy[predictors].corr()*100)
    corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape)).astype(bool))
    np.fill_diagonal(corr_mat.values, np.nan)
    corr_mat.to_csv(expl_dir+'predictors_corr_matrix.csv')
    
    corr_df = corr_mat.stack().reset_index()
    corr_df.columns = ['Feature_1', 'Feature_2', 'Corr']
    corr_df = corr_df.sort_values('Corr', ascending=False)
    corr_df.to_csv(expl_dir+'predictors_corr_table.csv', index=False)
    
    corr_to_predictand_df = pd.DataFrame(columns=predictands)
    for predictand in predictands:
        for predictor in predictors:
            corr_to_predictand_df.loc[predictor, predictand] = round(coefs_regr_df_copy[[predictor, predictand]].corr().iloc[0,1]*100, 1)
    corr_to_predictand_df.to_csv(expl_dir+'corr_to_predictand.csv')
    
    for predictand in predictands:
        # Handles folder creation/reset
        predictand_dir = expl_dir+'Scatterplots/'+predictand+'/'
        if os.path.exists(predictand_dir):
            shutil.rmtree(predictand_dir)
        os.makedirs(predictand_dir)
        
        # Plot scatter
        for predictor in predictors:
            # print(predictand + ' vs. ' + predictor)
            x, y = coefs_regr_df_copy[[predictor]].dropna(), coefs_regr_df_copy[predictand].dropna()
            x, y = x.loc[x.index[x.index.isin(y.index)]].values, y.loc[y.index[y.index.isin(x.index)]].values
            plt.figure(figsize=(7, 6))
            plt.scatter(coefs_regr_df[predictor], coefs_regr_df_copy[predictand])
            plt.plot(x, LinearRegression().fit(x, y).predict(x),color='firebrick')
            plt.title(predictor + ' vs. ' + predictand)
            plt.xlabel(predictor)
            plt.ylabel(predictand)
            plt.savefig(predictand_dir + predictor + '_vs_' + predictand + '.png')
            plt.close()
    
    # plt.figure(figsize=(7, 6))
    # plt.scatter(1/(coefs_regr_df['SKEW']), coefs_regr_df['sigma_MVP_fwd3mo_MA'])
    # plt.show()    
    
    # PCA
    coefs_pred_df = coefs_regr_df_copy[predictors].dropna()
    coefs_pred_df_scaled = StandardScaler().fit_transform(coefs_pred_df)
    pca = PCA()
    pc_coefs_pred_df = pca.fit_transform(coefs_pred_df_scaled)
    # pc_df = pd.DataFrame(data = pc_coefs_pred_df, columns = ['PC1', 'PC2'])
    
    pca_dir = expl_dir+'PCA/'
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)
    
    # Biplot
    def biplot(score,coeff,labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, c='lightskyblue')
        texts = []
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                texts.append(plt.text(coeff[i,0]* 1, coeff[i,1] * 1, "Var"+str(i+1), c='black', ha='center', va='center', size=15))
            else:
                texts.append(plt.text(coeff[i,0]* 1, coeff[i,1] * 1, labels[i], c='black', ha='center', va='center', size=15))
        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()        
        adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    plt.figure(figsize=(15, 13))
    biplot(pc_coefs_pred_df[:,0:2],np.transpose(pca.components_[0:2,:]), labels=coefs_pred_df.columns)
    plt.savefig(pca_dir + 'pca_biplot.png')
    plt.close()
    
    # Screeplot
    PC_values = np.arange(10) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_[:10], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.savefig(pca_dir +'pca_screeplot.png')
    plt.close()
    
    # Cumulative Screeplot
    PC_values = np.arange(10) + 1
    plt.plot(PC_values, np.cumsum(pca.explained_variance_ratio_[:10]), 'ro-', linewidth=2)
    plt.title('Cumulative Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.savefig(pca_dir + 'pca_cumulative_screeplot.png')
    plt.close()
    
    print('Exploratory Analysis Runtime: ' + str(round((time.time() - start_time)/60, 1)) + ' mins')
    return True
