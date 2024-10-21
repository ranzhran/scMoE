import argparse
import random,os,sys
import numpy as np
import csv
import gc
from scipy import stats
from collections import defaultdict
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, History
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr,spearmanr
from model import KerasMultiSourceGCNModel
import hickle as hkl
import pickle
import scipy.sparse as sp
import argparse
from tensorflow.keras.models import load_model



####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='1', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=False, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=False, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', default=False, type=bool, help='use GlobalMaxPooling for GCN') #drug graph average
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut,use_gexp,use_methy = args.use_mut,args.use_gexp, args.use_methy
israndom=args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' +GCN_deploy

####################################Constants Settings###########################
TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
DPATH = '../data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
Max_atoms = 100


def MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,filtered):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature0 = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    gexpr_feature1 = pd.read_csv('../data/scgpt_cell_emb.csv', index_col=0)
    gexpr_feature2 = pd.read_csv('../data/geneformer_cell_emb.csv', index_col=0)
   
    gexpr_np = np.load('../data/50M-0.1B-res_embedding.npy')
    gexpr_feature3 = pd.DataFrame(gexpr_np,index=gexpr_feature0.index)
    gexpr_feature4 = pd.read_csv('../data/scbert_cell_emb_0817.csv', header=0, index_col=0)
    gexpr_feature4.index=gexpr_feature0.index

    gexpr_feature5 = pd.read_csv('../data/CellPLM_cell_emb.csv', header=0, index_col=0)
    gexpr_feature6 = pd.read_csv('../data/CellLM_emb.csv', header=0, index_col=0)
    gexpr_feature7 = pd.read_csv('../data/scMulan_emb.csv', header=0, index_col=0)
    gexpr_feature8 = pd.read_csv('../data/genept_emb.csv', header=0, index_col=0)

    gexpr_feature = [gexpr_feature0, gexpr_feature1, gexpr_feature2, gexpr_feature3,gexpr_feature4,gexpr_feature5, gexpr_feature6,gexpr_feature7,gexpr_feature8]


    #load drug embedding
    with open('../data/smiles_gpt_embedding.pkl', 'rb') as f:
        drug_emb0 = pickle.load(f)
    with open('../data/KV-PLM_embedding.pkl', 'rb') as f:
        drug_emb1 = pickle.load(f)
    with open('../data/molecule_stm_embeddings.pkl', 'rb') as f:
        drug_emb2 = pickle.load(f)
    with open('../data/molLM_embeddings_dict.pkl', 'rb') as f:
        drug_emb3 = pickle.load(f)
    with open('../data/spmm_embedding.pkl', 'rb') as f:
        drug_emb4 = pickle.load(f)
    with open('../data/MolFM_embedding.pkl', 'rb') as f:
        drug_emb5 = pickle.load(f)
    with open('../data/molformer_embeddings.pkl', 'rb') as f:
        drug_emb6 = pickle.load(f)
    with open('../data/unimol_embedding.pkl', 'rb') as f:
        drug_emb7 = pickle.load(f)
    
    drug_emb = [drug_emb0,drug_emb1,drug_emb2,drug_emb3,drug_emb4,drug_emb5,drug_emb6, drug_emb7]
      
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature[0].index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    data_idx.append((each_cellline,pubchem_id,ln_IC50,cellline2cancertype[each_cellline])) 
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return drug_feature, drug_emb, gexpr_feature, data_idx


# drug cold start
def DataSplit(data_idx):
    data_train_idx, data_test_idx = [], []
    pubchem_id_dict = defaultdict(list)
    for item in data_idx:
        pubchem_id_dict[item[1]].append(item)
    pubchem_ids = list(pubchem_id_dict.keys())
    test_pubchem_ids = set(random.sample(pubchem_ids, int(0.2 * len(pubchem_ids))))

    for item in data_idx:
        if item[1] in test_pubchem_ids:
            data_test_idx.append(item)
        else:
            data_train_idx.append(item)

    return data_train_idx, data_test_idx


# cell line cold start
# def DataSplit(data_idx):
#     data_train_idx, data_test_idx = [], []
#     test_cell_type = set(random.sample(TCGA_label_set, int(0.2 * len(TCGA_label_set))))

#     for item in data_idx:
#         if item[-1] in test_cell_type:
#             data_test_idx.append(item)
#         else:
#             data_train_idx.append(item)

#     return data_train_idx, data_test_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def FeatureExtract(data_idx, drug_feature, drug_emb_list, gexpr_feature_list):
    cancer_type_list = []
    nb_instance = len(data_idx)

    nb_gexpr_features = [gexpr_feature.shape[1] for gexpr_feature in gexpr_feature_list]
    nb_drug_emb = [next(iter(drug_emb_dict.values())).shape[1] for drug_emb_dict in drug_emb_list]
    
    # Initialize lists to hold features
    drug_data = [[] for _ in range(nb_instance)]
    drug_emb_data = [[] for _ in range(nb_instance)]
    gexpr_data = [[] for _ in range(nb_instance)]
    target = np.zeros(nb_instance, dtype='float32')

    csv_data = []
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]

        # Modify
        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
        # Fill drug data, padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list)
        # Concatenate drug embeddings
        drug_emb_data[idx] = [np.squeeze(drug_emb_list[0][str(pubchem_id)]), np.squeeze(drug_emb_list[1][str(pubchem_id)].numpy()), np.squeeze(drug_emb_list[2][str(pubchem_id)]), np.squeeze(drug_emb_list[3][str(pubchem_id)]),np.squeeze(drug_emb_list[4][str(pubchem_id)]), np.squeeze(drug_emb_list[5][str(pubchem_id)]),np.squeeze(drug_emb_list[6][str(pubchem_id)]),np.squeeze(drug_emb_list[7][str(pubchem_id)])]
        
        
        gexpr_data[idx] = [gexpr_feature_list[0].loc[cell_line_id].values, gexpr_feature_list[1].loc[cell_line_id].values, gexpr_feature_list[2].loc[cell_line_id].values,gexpr_feature_list[3].loc[cell_line_id].values,gexpr_feature_list[4].loc[cell_line_id].values, gexpr_feature_list[5].loc[cell_line_id].values, gexpr_feature_list[6].loc[cell_line_id].values, gexpr_feature_list[7].loc[cell_line_id].values, gexpr_feature_list[8].loc[cell_line_id].values]

        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
        csv_data.append([pubchem_id, cell_line_id, ln_IC50])

    return drug_data, drug_emb_data, gexpr_data, target, cancer_type_list, nb_drug_emb, nb_gexpr_features



class MyCallback(Callback):
    def __init__(self,validation_data,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1][0]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('../checkpoint/1013-BestDeepCDR_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_ = self.model.predict(self.x_val)
        y_pred_val = y_pred_[0]


        pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
        scc_val = spearmanr(self.y_val, y_pred_val[:,0])[0]
        print("===========Epoch",epoch,"============")
        print('pcc-val: %s' % str(round(pcc_val, 4)))
        print('scc-val: %s' % str(round(scc_val, 4)))

        if pcc_val > self.best:
            self.best = pcc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        
        # new
        tf.keras.backend.clear_session()
        gc.collect()
        return


@tf.function
def info_nce_loss(expert_outputs):

    shape = tf.shape(expert_outputs)
    batch_size = shape[0]
    feature_dim = shape[1]
    n_experts = shape[2]

    expert_outputs = tf.transpose(expert_outputs, perm=[0, 2, 1])
    expert_outputs = tf.nn.l2_normalize(expert_outputs, axis=-1)

    positive_samples = tf.TensorArray(tf.float32, size=batch_size)
    negative_samples = tf.TensorArray(tf.float32, size=batch_size)

    for i in range(batch_size):
        idx_j = tf.random.uniform([], minval=0, maxval=n_experts, dtype=tf.int32)
        idx_k = tf.random.uniform([], minval=0, maxval=n_experts, dtype=tf.int32)

        expert_j = expert_outputs[i, idx_j]
        expert_k = expert_outputs[i, idx_k]
        pos_sim = tf.reduce_sum(expert_j * expert_k) / (tf.norm(expert_j) * tf.norm(expert_k))
        positive_samples = positive_samples.write(i, pos_sim)

        next_index = (i + 1) % batch_size
        neg_expert = tf.random.uniform([], minval=0, maxval=n_experts, dtype=tf.int32)

        neg_sim = tf.reduce_sum(expert_j * expert_outputs[next_index, neg_expert]) / (
            tf.norm(expert_j) * tf.norm(expert_outputs[next_index, neg_expert]))
        negative_samples = negative_samples.write(i, neg_sim)

    positive_samples = positive_samples.stack()
    negative_samples = negative_samples.stack()

    logits = tf.concat([positive_samples, negative_samples], axis=0)
    labels = tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits / 0.2)
    return tf.reduce_mean(loss)


def custom_loss1(y_true, y_pred):
    supervised_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return supervised_loss

def custom_loss2(y_true, y_pred):
    info_loss = info_nce_loss(y_pred)
    return info_loss

def custom_loss3(y_true, y_pred):
    info_loss = info_nce_loss(y_pred)
    return info_loss



def ModelTraining(model,X_drug_data_train,X_drug_emb_train,X_gexpr_data_train,Y_train,validation_data,nb_epoch=100):#100
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #lr=0.001

    model.compile(optimizer=optimizer, loss=[custom_loss1, custom_loss2, custom_loss3],  
              loss_weights=[1.0, 0.1, 0.1])

    # 0919
    # EarlyStopping(monitor='val_loss',patience=100)
    callbacks = [ModelCheckpoint('../checkpoint/best_DeepCDR_%s.h5'%model_suffix,monitor='val_loss',save_best_only=False, save_weights_only=False),
                MyCallback(validation_data=validation_data,patience=10)]
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    X_drug_emb0_train = [item[0] for item in X_drug_emb_train]
    X_drug_emb1_train = [item[1] for item in X_drug_emb_train]
    X_drug_emb2_train = [item[2] for item in X_drug_emb_train]
    X_drug_emb3_train = [item[3] for item in X_drug_emb_train]
    X_drug_emb4_train = [item[4] for item in X_drug_emb_train]
    X_drug_emb5_train = [item[5] for item in X_drug_emb_train]
    X_drug_emb6_train = [item[6] for item in X_drug_emb_train]
    X_drug_emb7_train = [item[7] for item in X_drug_emb_train]

    X_drug_emb0_train = np.array(X_drug_emb0_train)
    X_drug_emb1_train = np.array(X_drug_emb1_train)
    X_drug_emb2_train = np.array(X_drug_emb2_train)
    X_drug_emb3_train = np.array(X_drug_emb3_train)
    X_drug_emb4_train = np.array(X_drug_emb4_train)
    X_drug_emb5_train = np.array(X_drug_emb5_train)
    X_drug_emb6_train = np.array(X_drug_emb6_train)
    X_drug_emb7_train = np.array(X_drug_emb7_train)

    X_gexp0_train = [item[0] for item in X_gexpr_data_train]
    X_gexp1_train = [item[1] for item in X_gexpr_data_train]
    X_gexp2_train = [item[2] for item in X_gexpr_data_train]
    X_gexp3_train = [item[3] for item in X_gexpr_data_train]
    X_gexp4_train = [item[4] for item in X_gexpr_data_train]
    X_gexp5_train = [item[5] for item in X_gexpr_data_train]
    X_gexp6_train = [item[6] for item in X_gexpr_data_train]
    X_gexp7_train = [item[7] for item in X_gexpr_data_train]
    X_gexp8_train = [item[8] for item in X_gexpr_data_train]

    X_gexp0_train = np.array(X_gexp0_train)
    X_gexp1_train = np.array(X_gexp1_train)
    X_gexp2_train = np.array(X_gexp2_train)
    X_gexp3_train = np.array(X_gexp3_train)
    X_gexp4_train = np.array(X_gexp4_train)
    X_gexp5_train = np.array(X_gexp5_train)
    X_gexp6_train = np.array(X_gexp6_train)
    X_gexp7_train = np.array(X_gexp7_train)
    X_gexp8_train = np.array(X_gexp8_train)

    model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_drug_emb0_train,X_drug_emb1_train,X_drug_emb2_train, X_drug_emb3_train, X_drug_emb4_train, X_drug_emb5_train,X_drug_emb6_train, X_drug_emb7_train,X_gexp0_train,X_gexp1_train,X_gexp2_train,X_gexp3_train,X_gexp4_train,X_gexp5_train, X_gexp6_train, X_gexp7_train, X_gexp8_train],y=[Y_train,Y_train, Y_train],batch_size=64,epochs=nb_epoch,validation_split=0,callbacks=callbacks,verbose=0) #64
    return model


def ModelEvaluate(model,X_drug_data_test,X_drug_emb_test,X_gexpr_data_test,Y_test,cancer_type_test_list,file_path):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    

    X_drug_emb0_test = [item[0] for item in X_drug_emb_test]
    X_drug_emb1_test = [item[1] for item in X_drug_emb_test]
    X_drug_emb2_test = [item[2] for item in X_drug_emb_test]
    X_drug_emb3_test = [item[3] for item in X_drug_emb_test]
    X_drug_emb4_test = [item[4] for item in X_drug_emb_test]
    X_drug_emb5_test = [item[5] for item in X_drug_emb_test]
    X_drug_emb6_test = [item[6] for item in X_drug_emb_test]
    X_drug_emb7_test = [item[7] for item in X_drug_emb_test]

    X_drug_emb0_test = np.array(X_drug_emb0_test)
    X_drug_emb1_test = np.array(X_drug_emb1_test)
    X_drug_emb2_test = np.array(X_drug_emb2_test)
    X_drug_emb3_test = np.array(X_drug_emb3_test)
    X_drug_emb4_test = np.array(X_drug_emb4_test)
    X_drug_emb5_test = np.array(X_drug_emb5_test)
    X_drug_emb6_test = np.array(X_drug_emb6_test)
    X_drug_emb7_test = np.array(X_drug_emb7_test)


    X_gexp0_test = [item[0] for item in X_gexpr_data_test]
    X_gexp1_test = [item[1] for item in X_gexpr_data_test]
    X_gexp2_test = [item[2] for item in X_gexpr_data_test]
    X_gexp3_test = [item[3] for item in X_gexpr_data_test]
    X_gexp4_test = [item[4] for item in X_gexpr_data_test]
    X_gexp5_test = [item[5] for item in X_gexpr_data_test]
    X_gexp6_test = [item[6] for item in X_gexpr_data_test]
    X_gexp7_test = [item[7] for item in X_gexpr_data_test]
    X_gexp8_test = [item[8] for item in X_gexpr_data_test]

    X_gexp0_test = np.array(X_gexp0_test)
    X_gexp1_test = np.array(X_gexp1_test)
    X_gexp2_test = np.array(X_gexp2_test)
    X_gexp3_test = np.array(X_gexp3_test)
    X_gexp4_test = np.array(X_gexp4_test)
    X_gexp5_test = np.array(X_gexp5_test)
    X_gexp6_test = np.array(X_gexp6_test)
    X_gexp7_test = np.array(X_gexp7_test)
    X_gexp8_test = np.array(X_gexp8_test)

    Y_pred_ = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_drug_emb0_test, X_drug_emb1_test, X_drug_emb2_test, X_drug_emb3_test, X_drug_emb4_test, X_drug_emb5_test, X_drug_emb6_test, X_drug_emb7_test,X_gexp0_test, X_gexp1_test, X_gexp2_test, X_gexp3_test, X_gexp4_test, X_gexp5_test, X_gexp6_test, X_gexp7_test, X_gexp8_test])
    Y_pred = Y_pred_[0]
    overall_pcc = pearsonr(Y_pred[:,0],Y_test)[0]
    print("The overall Pearson's correlation is %.4f."%overall_pcc)
    overall_scc = spearmanr(Y_pred[:,0],Y_test)[0]
    print("The overall Spearman's correlation is %.4f."%overall_scc)
    

def main():
    drug_feature,drug_emb, gexpr_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,False)
    data_train_idx,data_test_idx = DataSplit(data_idx)
    #Extract features for training and test 
    X_drug_data_train,X_drug_emb_train,X_gexpr_data_train,Y_train,cancer_type_train_list, nb_drug_emb_train, nb_gexp_train  = FeatureExtract(data_train_idx,drug_feature,drug_emb,gexpr_feature)
    X_drug_data_test,X_drug_emb_test,X_gexpr_data_test,Y_test,cancer_type_test_list, nb_drug_emb_test, nb_gexp_test = FeatureExtract(data_test_idx,drug_feature,drug_emb,gexpr_feature)

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    X_drug_emb0_test = [item[0] for item in X_drug_emb_test]
    X_drug_emb1_test = [item[1] for item in X_drug_emb_test]
    X_drug_emb2_test = [item[2] for item in X_drug_emb_test]
    X_drug_emb3_test = [item[3] for item in X_drug_emb_test]
    X_drug_emb4_test = [item[4] for item in X_drug_emb_test]
    X_drug_emb5_test = [item[5] for item in X_drug_emb_test]
    X_drug_emb6_test = [item[6] for item in X_drug_emb_test]
    X_drug_emb7_test = [item[7] for item in X_drug_emb_test]

    X_drug_emb0_test = np.array(X_drug_emb0_test)
    X_drug_emb1_test = np.array(X_drug_emb1_test)
    X_drug_emb2_test = np.array(X_drug_emb2_test)
    X_drug_emb3_test = np.array(X_drug_emb3_test)
    X_drug_emb4_test = np.array(X_drug_emb4_test)
    X_drug_emb5_test = np.array(X_drug_emb5_test)
    X_drug_emb6_test = np.array(X_drug_emb6_test)
    X_drug_emb7_test = np.array(X_drug_emb7_test)


    X_gexp0_test = [item[0] for item in X_gexpr_data_test]
    X_gexp1_test = [item[1] for item in X_gexpr_data_test]
    X_gexp2_test = [item[2] for item in X_gexpr_data_test]
    X_gexp3_test = [item[3] for item in X_gexpr_data_test]
    X_gexp4_test = [item[4] for item in X_gexpr_data_test]
    X_gexp5_test = [item[5] for item in X_gexpr_data_test]
    X_gexp6_test = [item[6] for item in X_gexpr_data_test]
    X_gexp7_test = [item[7] for item in X_gexpr_data_test]
    X_gexp8_test = [item[8] for item in X_gexpr_data_test]

    X_gexp0_test = np.array(X_gexp0_test)
    X_gexp1_test = np.array(X_gexp1_test)
    X_gexp2_test = np.array(X_gexp2_test)
    X_gexp3_test = np.array(X_gexp3_test)
    X_gexp4_test = np.array(X_gexp4_test)
    X_gexp5_test = np.array(X_gexp5_test)
    X_gexp6_test = np.array(X_gexp6_test)
    X_gexp7_test = np.array(X_gexp7_test)
    X_gexp8_test = np.array(X_gexp8_test)
    
    validation_data = [[X_drug_feat_data_test,X_drug_adj_data_test,X_drug_emb0_test, X_drug_emb1_test, X_drug_emb2_test, X_drug_emb3_test, X_drug_emb4_test, X_drug_emb5_test,X_drug_emb6_test, X_drug_emb7_test,X_gexp0_test, X_gexp1_test, X_gexp2_test, X_gexp3_test, X_gexp4_test, X_gexp5_test, X_gexp6_test, X_gexp7_test, X_gexp8_test],[Y_test,Y_test,Y_test]]
    model = KerasMultiSourceGCNModel(use_mut,use_gexp,use_methy).createMaster(X_drug_data_train[0][0].shape[-1],nb_drug_emb_train,nb_gexp_train,args.unit_list,args.use_relu,args.use_bn,args.use_GMP)

    print('Begin training...')
    model = ModelTraining(model,X_drug_data_train,X_drug_emb_train,X_gexpr_data_train,Y_train,validation_data,nb_epoch=100)
    ModelEvaluate(model,X_drug_data_test,X_drug_emb_test,X_gexpr_data_test,Y_test,cancer_type_test_list,'%s/DeepCDR_%s.log'%(DPATH,model_suffix))

if __name__=='__main__':
    # 5-fold cross validation
    for i in range (5):
        random.seed(i)
        main()
    
