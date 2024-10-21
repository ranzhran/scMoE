import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

import os
# import cpa
from cpa._model import CPA
import scanpy as sc
import statistics
import numpy as np
import ot

def compute_wasserstein_distance(X1, X2):

    a = np.ones((X1.shape[0],)) / X1.shape[0]  # X1 的每个样本点的权重
    b = np.ones((X2.shape[0],)) / X2.shape[0]  # X2 的每个样本点的权重

    cost_matrix = ot.dist(X1, X2)

    emd2_distance = ot.emd2(a, b, cost_matrix)

    return emd2_distance


def compute_metrics(pred1, true1, deg_list, n_top_degs):
    for n_top_deg in n_top_degs:
        if n_top_deg is not None:
            # 获取前 n_top_deg 的基因索引
            degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            # 获取所有基因的索引
            degs = np.arange(true1.shape[1])
            n_top_deg = 'all'

        # 根据索引提取预测和真实的基因表达
        x_true_deg = true1[:, degs]
        x_pred_deg = pred1[:, degs]

        # 计算 Wasserstein 距离
        wasserstein_distance = compute_wasserstein_distance(x_pred_deg, x_true_deg)

        # 计算 Spearman 和 Pearson 相关系数
        scc_mean_deg, _ = spearmanr(x_true_deg.mean(0), x_pred_deg.mean(0))
        pcc_mean_deg, _ = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))

        # 计算 R² 和 MSE
        r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
        r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))
        mse = mean_squared_error(x_true_deg.mean(0), x_pred_deg.mean(0))

        # 输出结果
        print("n_top_deg:", n_top_deg)
        print("Wasserstein Distance:", wasserstein_distance)
        print("Spearman Correlation:", scc_mean_deg)
        print("Pearson Correlation:", pcc_mean_deg)
        print("R² mean:", r2_mean_deg)
        print("R² var:", r2_var_deg)
        print("Mean Squared Error (MSE):", mse)



data_path = '/home/zhangran/scdrug/cpa/datasets/combo_sciplex_prep_hvg_filtered.h5ad'

adata = sc.read(data_path)

adata.X = adata.layers['counts'].copy()

n_top_degs = [2, 3, 5, 10, 20, 30, 40, 50, None]
    
# test1
print('test:CHEMBL1213492+CHEMBL491473========================================')
test1 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true1 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL1213492+CHEMBL491473_3.0+3.0'].copy()
# x_pred1 = adata[adata.obs.split_1ct_MEC == 'train'].copy()
x_pred1 = adata[adata.obs.split_1ct_MEC != 'ood'].copy()
pred1 = x_pred1.layers['counts'].toarray()
true1 = x_true1.layers['counts'].toarray()
true1 = np.log1p(true1)
pred1 = np.log1p(pred1)

deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL1213492+CHEMBL491473_3.0+3.0']
compute_metrics(pred1, true1, deg_list, n_top_degs)


# test2
print('test:CHEMBL483254+CHEMBL4297436========================================')
test2 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true2 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL483254+CHEMBL4297436_3.0+3.0'].copy() #14,16
# x_pred2 = adata[adata.obs.split_1ct_MEC == 'train'].copy()
x_pred2 = adata[adata.obs.split_1ct_MEC != 'ood'].copy()
pred2 = x_pred2.layers['counts'].toarray()
true2 = x_true2.layers['counts'].toarray()
true2 = np.log1p(true2)
pred2 = np.log1p(pred2)

deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL483254+CHEMBL4297436_3.0+3.0']
compute_metrics(pred2, true2, deg_list, n_top_degs)

# test3
print('test:CHEMBL356066+CHEMBL402548========================================')
test3 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true3 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL356066+CHEMBL402548_3.0+3.0'].copy() #11,13
# x_pred3 = adata[adata.obs.split_1ct_MEC == 'train'].copy()
x_pred3 = adata[adata.obs.split_1ct_MEC != 'ood'].copy()
pred3 = x_pred3.layers['counts'].toarray()
true3 = x_true3.layers['counts'].toarray()
true3 = np.log1p(true3)
pred3 = np.log1p(pred3)

deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL356066+CHEMBL402548_3.0+3.0']
compute_metrics(pred3, true3, deg_list, n_top_degs)

# test4
print('test:CHEMBL483254+CHEMBL383824========================================')
test4 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true4 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL483254+CHEMBL383824_3.0+3.0'].copy() #12,16
# x_pred4 = adata[adata.obs.split_1ct_MEC == 'train'].copy()
x_pred4 = adata[adata.obs.split_1ct_MEC != 'ood'].copy()
pred4 = x_pred4.layers['counts'].toarray()
true4 = x_true4.layers['counts'].toarray()
true4 = np.log1p(true4)
pred4 = np.log1p(pred4)

deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL483254+CHEMBL383824_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]

x_true_deg4 = true4[:, degs]
x_pred_deg4 = pred4[:, degs]
print("x_pred_deg4",  x_pred_deg4.mean(0))
print("x_true_deg4",  x_true_deg4.mean(0))
compute_metrics(pred4, true4, deg_list, n_top_degs)

# test5
print('test:CHEMBL4297436+CHEMBL383824========================================')
test5 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true5 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL4297436+CHEMBL383824_3.0+3.0'].copy() #12,14
# x_pred5 = adata[adata.obs.split_1ct_MEC == 'train'].copy()
x_pred5 = adata[adata.obs.split_1ct_MEC != 'ood'].copy()
pred5 = x_pred5.layers['counts'].toarray()
true5 = x_true5.layers['counts'].toarray()
true5 = np.log1p(true5)
pred5 = np.log1p(pred5)
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL4297436+CHEMBL383824_3.0+3.0']
compute_metrics(pred5, true5, deg_list, n_top_degs)



