import sys
from sklearn.metrics import r2_score
import numpy as np

import os
from cpa._model import CPA
import scanpy as sc

sc.settings.set_figure_params(dpi=100)

data_path = '/home/zhangran/scdrug/cpa/datasets/combo_sciplex_prep_hvg_filtered.h5ad'

adata = sc.read(data_path)



adata.X = adata.layers['counts'].copy()

adata.obs['split_1ct_MEC'].value_counts()
ood_conds = list(adata[adata.obs['split_1ct_MEC'] == 'ood'].obs['condition_ID'].value_counts().index)
adata.obs['condition_split'] = adata.obs['condition_ID'].apply(lambda x: x if x in ood_conds else 'other')
# sc.settings.verbosity = 3
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)

CPA.setup_anndata(adata,
                      perturbation_key='condition_ID',
                      dosage_key='log_dose',
                      control_group='CHEMBL504',
                      batch_key=None,
                      smiles_key='smiles_rdkit',
                      is_count_data=True,
                      categorical_covariate_keys=['cell_type'],
                      deg_uns_key='rank_genes_groups_cov',
                      deg_uns_cat_key='cov_drug_dose',
                      max_comb_len=2,
                     )


print('=======end setup adata')

ae_hparams = {'n_latent': 64,
 'recon_loss': 'nb',
 'doser_type': 'linear',
 'n_hidden_encoder': 256,
 'n_layers_encoder': 3,
 'n_hidden_decoder': 512,
 'n_layers_decoder': 2,
 'use_batch_norm_encoder': True,
 'use_layer_norm_encoder': False,
 'use_batch_norm_decoder': True,
 'use_layer_norm_decoder': False,
 'dropout_rate_encoder': 0.25,
 'dropout_rate_decoder': 0.25,
 'variational': False,
 'seed': 6478}

trainer_params = {'n_epochs_kl_warmup': None,
 'n_epochs_pretrain_ae': 50,
 'n_epochs_adv_warmup': 100,
 'n_epochs_mixup_warmup': 10,
 'mixup_alpha': 0.1,
 'adv_steps': None,
 'n_hidden_adv': 128,
 'n_layers_adv': 3,
 'use_batch_norm_adv': False,
 'use_layer_norm_adv': False,
 'dropout_rate_adv': 0.2,
 'reg_adv': 10.0,
 'pen_adv': 0.1,
 'lr': 0.0003,
 'wd': 4e-07,
 'adv_lr': 0.0003,
 'adv_wd': 4e-07,
 'adv_loss': 'cce',
 'doser_lr': 0.0003,
 'doser_wd': 4e-07,
 'do_clip_grad': False,
 'gradient_clip_value': 1.0,
 'step_size_lr': 10}


model = CPA(adata=adata,
                split_key='split_1ct_MEC',
                train_split='train',
                valid_split='valid',
                test_split='ood',
                use_rdkit_embeddings=True,
                **ae_hparams,
               )

model.train(max_epochs=100, #2000
            use_gpu=True,
            batch_size=1024, #128 #512
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path='/home/zhangran/scdrug/cpa/lightning_logs/combo/cpa_drug/',
           )

model.predict(adata, batch_size=1024)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm

n_top_degs = [10, 20, 50, None] # None means all genes

results = defaultdict(list)
ctrl_adata = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
for cat in tqdm(adata.obs['cov_drug_dose'].unique()):
    if 'CHEMBL504' not in cat:
        cat_adata = adata[adata.obs['cov_drug_dose'] == cat].copy()

        deg_cat = f'{cat}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

        x_true = cat_adata.layers['counts'].toarray()
        x_pred = cat_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()

        x_true = np.log1p(x_true)
        x_pred = np.log1p(x_pred)
        x_ctrl = np.log1p(x_ctrl)

        for n_top_deg in n_top_degs:
            if n_top_deg is not None:
                degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
            else:
                degs = np.arange(adata.n_vars)
                n_top_deg = 'all'

            x_true_deg = x_true[:, degs]
            x_pred_deg = x_pred[:, degs]
            x_ctrl_deg = x_ctrl[:, degs]

            r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
            r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))

            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

            cov, cond, dose = cat.split('_')

            results['cell_type'].append(cov)
            results['condition'].append(cond)
            results['dose'].append(dose)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_var_deg'].append(r2_var_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
            results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

df = pd.DataFrame(results)
df.to_csv('logs0820/results/cpa_drug_base.csv', index=True) 