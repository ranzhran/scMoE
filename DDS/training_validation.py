import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from models.gat_gcn_test import GAT_GCN
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output, info_loss = model(data1, data2)
        loss = loss_fn(output, y) + 0.1*info_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {}  Loss: {:.6f}'.format(epoch, loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output,_ = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()



for fold in range(5):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print('============FOLD ', fold+1, '===============')

    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LR = 0.001
    LOG_INTERVAL = 20
    NUM_EPOCHS = 100

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    file = ['new_labels_0_10']

    # CPU or GPU

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    for datafile in file:


        independent_drug1_data = TestbedDataset(root='data', dataset='independent_input_drug1')
        independent_drug2_data = TestbedDataset(root='data', dataset='independent_input_drug2')

        lenth = len(independent_drug1_data)
        drug1_data = TestbedDataset(root='data', dataset='new_labels_0_10_drug1')
        drug2_data = TestbedDataset(root='data', dataset='new_labels_0_10_drug2')
        lenth = len(drug1_data)
        random_num = random.sample(range(0, lenth), lenth)
        drug1_data = drug1_data[random_num]
        drug2_data = drug2_data[random_num]

        drug1_loader_train = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
        drug2_loader_train = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

        independent_drug1_loader_test = DataLoader(independent_drug1_data, batch_size=TEST_BATCH_SIZE, shuffle=None)
        independent_drug2_loader_test = DataLoader(independent_drug2_data, batch_size=TEST_BATCH_SIZE, shuffle=None)

        modeling = GAT_GCN
        model = modeling().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)


        model_file_name = 'data/result/' + datafile + 'GAT_GCN--model.pt'

        file_AUCs = 'data/result/' + datafile + 'GAT_GCN--AUCs.txt'
        AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')
        
        best_auc = 0
        best_res = []

        for epoch in range(NUM_EPOCHS):
            train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)

            if (epoch+1) % 1 == 0:

                T, S, Y = predicting(model, device, independent_drug1_loader_test, independent_drug2_loader_test)
                # torch.save(model.state_dict(), model_file_name)
                independent_num_file_name = 'data/result/new_labels_0_10_leave/' + datafile + '--result' + str(epoch) + '.csv'
                independent_num = []
                # independent_num.append(test_num)
                # independent_num.append(T)
                independent_num.append(Y)
                independent_num.append(S)
                txtDF = pd.DataFrame(data=list(map(list, zip(*independent_num))))
                txtDF.to_csv(independent_num_file_name, index=False, header=False)

                # T is correct label
                # S is predict score
                # Y is predict label

                # compute preformence
                AUC = roc_auc_score(T, S)
                precision, recall, threshold = metrics.precision_recall_curve(T, S)
                AUPR = metrics.auc(recall, precision)
                BACC = balanced_accuracy_score(T, Y)
                tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
                TPR = tp / (tp + fn)
                PREC = precision_score(T, Y)
                ACC = accuracy_score(T, Y)
                KAPPA = cohen_kappa_score(T, Y)

                AUCs = [epoch, AUC, AUPR, BACC, KAPPA]
                print(AUCs)
                
                if best_auc < AUC:
                    best_auc = AUC
                    best_res = AUCs                
                    save_AUCs(AUCs, file_AUCs)
                torch.save(model.state_dict(), model_file_name)
        print("BEST_AUC", best_auc)
        print("BEST_RES", best_res)

    print('============FOLD ', fold+1, ' END! ===============')

