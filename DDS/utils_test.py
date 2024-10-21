import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pickle
import csv

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)

        drug_path = './data/drug/'

        # 定义文件名列表
        embs = [
            'kv',
            'minimol',
            'molfm',
            'mollm',
            'smilesgpt',
            'spmm',
            'stm',
            'unimol'
        ]
        print("load drug===============")

        with open(drug_path + 'kv.pkl', 'rb') as f:
            kv = pickle.load(f)
        with open(drug_path + 'minimol.pkl', 'rb') as f:
            minimol = pickle.load(f)
        with open(drug_path + 'molfm.pkl', 'rb') as f:
            molfm = pickle.load(f)
        with open(drug_path + 'mollm.pkl', 'rb') as f:
            mollm = pickle.load(f)
        with open(drug_path + 'smilesgpt.pkl', 'rb') as f:
            smilesgpt = pickle.load(f)
        with open(drug_path + 'spmm.pkl', 'rb') as f:
            spmm = pickle.load(f)
        with open(drug_path + 'stm.pkl', 'rb') as f:
            stm = pickle.load(f)
        with open(drug_path + 'unimol.pkl', 'rb') as f:
            unimol = pickle.load(f)

        print("load drug finished===============")
        print("load celllm===============")

        cell_path = '/home/zhangran/scdrug/DeepDDS/data/cell/ind-'
        celllm = []
        with open(cell_path + 'celllm.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                celllm.append(row)
        celllm = np.array(celllm)

        print("load cellplm===============")

        cellplm = []
        with open(cell_path + 'cellplm.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                cellplm.append(row)
        cellplm = np.array(cellplm)
        
        print("load geneformer===============")
        geneformer = []
        with open(cell_path + 'geneformer.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                geneformer.append(row)
        geneformer = np.array(geneformer)

        print("load genept===============")
        genept = []
        with open(cell_path + 'genept.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                genept.append(row)
        genept = np.array(genept)

        print("load scbert===============")
        scbert = []
        with open(cell_path + 'scbert.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                scbert.append(row)
        scbert = np.array(scbert)

        print("load scf===============")
        scf = []
        with open(cell_path + 'scf.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                scf.append(row)
        scf = np.array(scf)

        print("load scgpt===============")
        scgpt = []
        with open(cell_path + 'scgpt.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                scgpt.append(row)
        scgpt = np.array(scgpt)

        print("load scmulan===============")
        scmulan = []
        with open(cell_path + 'scmulan.csv') as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:
                scmulan.append(row)
        scmulan = np.array(scmulan)

        print("begin looping===============")
        for i in range(data_len):
            print(i)
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]))
            
            cell = self.get_cell_feature(target, xt_featrue)
            c_celllm = self.get_cell_feature(target, celllm)
            c_cellplm = self.get_cell_feature(target, cellplm)
            c_geneformer = self.get_cell_feature(target, geneformer)
            c_genept = self.get_cell_feature(target, genept)
            c_scbert = self.get_cell_feature(target, scbert)
            c_scf = self.get_cell_feature(target, scf)
            c_scgpt = self.get_cell_feature(target, scgpt)
            c_scmulan = self.get_cell_feature(target, scmulan)


            if cell == False : # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])

            celllm_cell = []
            for n in c_celllm:
                celllm_cell.append(float(n))
            GCNData.celllm = torch.FloatTensor([celllm_cell])

            cellplm_cell = []
            for n in c_cellplm:
                cellplm_cell.append(float(n))
            GCNData.cellplm = torch.FloatTensor([cellplm_cell])

            geneformer_cell = []
            for n in c_geneformer:
                geneformer_cell.append(float(n))      
            GCNData.geneformer = torch.FloatTensor([geneformer_cell])

            genept_cell = []
            for n in c_genept:
                genept_cell.append(float(n))     
            GCNData.genept = torch.FloatTensor([genept_cell])

            scbert_cell = []
            for n in c_scbert:
                scbert_cell.append(float(n))   
            GCNData.scbert = torch.FloatTensor([scbert_cell])

            scf_cell = []
            for n in c_scf:
                scf_cell.append(float(n)) 
            GCNData.scf = torch.FloatTensor([scf_cell])

            scgpt_cell = []
            for n in c_scgpt:
                scgpt_cell.append(float(n)) 
            GCNData.scgpt = torch.FloatTensor([scgpt_cell])

            scmulan_cell = []
            for n in c_scmulan:
                scmulan_cell.append(float(n)) 
            GCNData.scmulan = torch.FloatTensor([scmulan_cell])


            GCNData.kv = torch.FloatTensor(kv[smiles])
            GCNData.minimol = torch.FloatTensor(minimol[smiles])
            GCNData.molfm = torch.FloatTensor(molfm[smiles])
            GCNData.mollm = torch.FloatTensor(mollm[smiles])
            GCNData.smilesgpt = torch.FloatTensor(smilesgpt[smiles])
            GCNData.spmm = torch.FloatTensor(spmm[smiles])
            GCNData.stm = torch.FloatTensor(stm[smiles])
            GCNData.unimol = torch.FloatTensor(unimol[smiles])


            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci