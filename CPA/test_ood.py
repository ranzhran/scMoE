import sys
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd

import os
from cpa._model import CPA
import scanpy as sc
import statistics
import numpy as np
import ot

def compute_wasserstein_distance(X1, X2):

    a = np.ones((X1.shape[0],)) / X1.shape[0] 
    b = np.ones((X2.shape[0],)) / X2.shape[0] 

    cost_matrix = ot.dist(X1, X2)

    emd2_distance = ot.emd2(a, b, cost_matrix)

    return emd2_distance


def compute_metrics(pred1, true1, deg_list, n_top_degs):
    for n_top_deg in n_top_degs:
        if n_top_deg is not None:
            degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            degs = np.arange(true1.shape[1])
            n_top_deg = 'all'

        x_true_deg = true1[:, degs]
        x_pred_deg = pred1[:, degs]

        scc_mean_deg, _ = spearmanr(x_true_deg.mean(0), x_pred_deg.mean(0))
        pcc_mean_deg, _ = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))

        print("n_top_deg:", n_top_deg)
        print("Spearman Correlation:", scc_mean_deg)
        print("Pearson Correlation:", pcc_mean_deg)




data_path = '/home/zhangran/scdrug/cpa/datasets/combo_sciplex_prep_hvg_filtered.h5ad'

adata = sc.read(data_path)

adata.X = adata.layers['counts'].copy()


scemb_path = '/home/zhangran/scdrug/cpa-cell/datasets/cell/'

file_names = [
    'celllm.csv',
    'cellplm.csv',
    'geneformer.csv',
    'genept.csv',
    'scf.csv',
    'scgpt.csv',
    'scmulan.csv',
    'scbert.csv'
]

sc_dim = []
sc_list = []

for file_name in file_names:
    file_path = scemb_path + file_name
    gexpr_feature = pd.read_csv(file_path, sep=',', header=0, index_col=[0])
    emb = gexpr_feature.to_numpy()
    sc_list.append(emb)
    dim = emb.shape[1]
    sc_dim.append(dim)

sc_embs = np.concatenate(sc_list, axis=1)
adata.obsm['emb'] = sc_embs


CPA.setup_anndata(adata,
                    perturbation_key='condition_ID',
                    dosage_key='log_dose',
                    control_group='CHEMBL504',
                    batch_key=None,
                    # smiles_key='smiles_rdkit',
                    is_count_data=True,
                    categorical_covariate_keys=['cell_type'],
                    deg_uns_key='rank_genes_groups_cov',
                    deg_uns_cat_key='cov_drug_dose',
                    max_comb_len=2,
                    )
ae_hparams = {
    "n_latent": 128,
    "recon_loss": "nb",
    "doser_type": "logsigm",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 3,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 3,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 434,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 3,
    "mixup_alpha": 0.1,
    "adv_steps": 2,
    "n_hidden_adv": 64,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 20.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 45,
}

adata.obs['split_1ct_MEC'].value_counts()
model = CPA(adata=adata,
                split_key='split_1ct_MEC',
                train_split='train',
                valid_split='valid',
                test_split='ood',
                **ae_hparams,
            )

model.train(max_epochs=2000,
            use_gpu=True,
            batch_size=128,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path='./log/',
        )

n_top_degs = [2, 3, 5, 10, 20, 30, 40, 50, None]

# test1
print('test:CHEMBL1213492+CHEMBL491473========================================')
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL1213492+CHEMBL491473_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]
deg_names = adata.var_names[degs]
test1 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true1 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL1213492+CHEMBL491473_3.0+3.0'].copy()
pred1 = model.custom_predict(covars_to_add=[[6, 17], [3.0, 3.0]], adata=test1, batch_size=128) #1024
true1 = x_true1.layers['counts'].toarray()

true1 = np.log1p(true1)
pred1 = np.log1p(pred1)

x_true_deg1 = true1[:, degs]
x_pred_deg1 = pred1[:, degs]
print("x_pred_deg1",  x_pred_deg1.mean(0))
print("x_true_deg1",  x_true_deg1.mean(0))
compute_metrics(pred1, true1, deg_list, n_top_degs)

# test2
print('test:CHEMBL483254+CHEMBL4297436========================================')
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL483254+CHEMBL4297436_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]
deg_names = adata.var_names[degs]
test2 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true2 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL483254+CHEMBL4297436_3.0+3.0'].copy() #14,16
pred2 = model.custom_predict(covars_to_add=[[14, 16], [3.0, 3.0]], adata=test2, batch_size=128)
true2 = x_true2.layers['counts'].toarray()

true2 = np.log1p(true2)
pred2 = np.log1p(pred2)

x_true_deg2 = true2[:, degs]
x_pred_deg2 = pred2[:, degs]
print("x_pred_deg2",  x_pred_deg2.mean(0))
print("x_true_deg2",  x_true_deg2.mean(0))
compute_metrics(pred2, true2, deg_list, n_top_degs)

# test3
print('test:CHEMBL356066+CHEMBL402548========================================')
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL356066+CHEMBL402548_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]
deg_names = adata.var_names[degs]
test3 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true3 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL356066+CHEMBL402548_3.0+3.0'].copy() #11,13
pred3 = model.custom_predict(covars_to_add=[[11, 13], [3.0, 3.0]], adata=test3, batch_size=128)
true3 = x_true3.layers['counts'].toarray()

true3 = np.log1p(true3)
pred3 = np.log1p(pred3)

x_true_deg3 = true3[:, degs]
x_pred_deg3 = pred3[:, degs]
print("x_pred_deg3",  x_pred_deg3.mean(0))
print("x_true_deg3",  x_true_deg3.mean(0))

compute_metrics(pred3, true3, deg_list, n_top_degs)

# test4
print('test:CHEMBL483254+CHEMBL383824========================================')
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL483254+CHEMBL383824_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]
deg_names = adata.var_names[degs]
test4 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true4 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL483254+CHEMBL383824_3.0+3.0'].copy() #12,16
pred4 = model.custom_predict(covars_to_add=[[12, 16], [3.0, 3.0]], adata=test4, batch_size=128)
true4 = x_true4.layers['counts'].toarray()

true4 = np.log1p(true4)
pred4 = np.log1p(pred4)

x_true_deg4 = true4[:, degs]
x_pred_deg4 = pred4[:, degs]
print("x_pred_deg4",  x_pred_deg4.mean(0))
print("x_true_deg4",  x_true_deg4.mean(0))
compute_metrics(pred4, true4, deg_list, n_top_degs)

# test5
print('test:CHEMBL4297436+CHEMBL383824========================================')
deg_list = adata.uns['rank_genes_groups_cov']['A549_CHEMBL4297436+CHEMBL383824_3.0+3.0']
degs = np.where(np.isin(adata.var_names, deg_list[:50]))[0]
deg_names = adata.var_names[degs]
test5 = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
x_true5 = adata[adata.obs['cov_drug_dose'] == 'A549_CHEMBL4297436+CHEMBL383824_3.0+3.0'].copy() #12,14
pred5 = model.custom_predict(covars_to_add=[[12, 14], [3.0, 3.0]], adata=test5, batch_size=128)
true5 = x_true5.layers['counts'].toarray()

true5 = np.log1p(true5)
pred5 = np.log1p(pred5)

x_true_deg5 = true5[:, degs]
x_pred_deg5 = pred5[:, degs]
print("x_pred_deg5",  x_pred_deg5.mean(0))
print("x_true_deg5",  x_true_deg5.mean(0)) 
compute_metrics(pred5, true5, deg_list, n_top_degs)
    


