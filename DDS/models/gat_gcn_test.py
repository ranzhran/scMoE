import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

def infonce_loss_cosine(expert_outputs, temperature=0.2):
    batch_size, feature_dim, n_experts = expert_outputs.shape
    
    expert_outputs = expert_outputs.permute(0, 2, 1)  
    expert_outputs = F.normalize(expert_outputs, dim=-1) 

    positive_samples = torch.zeros(batch_size, device=expert_outputs.device)
    negative_samples = torch.zeros(batch_size, device=expert_outputs.device)

    for i in range(batch_size):
        j, k = torch.randint(0, n_experts, (2,)).tolist()
        positive_samples[i] = F.cosine_similarity(
            expert_outputs[i, j],
            expert_outputs[i, k], 
            dim=-1
        )
        
        next_index = (i + 1) % batch_size
        neg_expert = torch.randint(0, n_experts, (1,)).item()
        negative_samples[i] = F.cosine_similarity(
            expert_outputs[i, j],
            expert_outputs[next_index, neg_expert],
            dim=-1
        )
    
    logits = torch.cat([positive_samples, negative_samples], dim=0)

    labels = torch.zeros(logits.shape[0], dtype=torch.float, device=expert_outputs.device)
    labels[:batch_size] = 1  

    loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)

    return loss


class MulMoE(nn.Module):
    def __init__(self, units, n_experts, expert_input_dims, feature_dim, n_heads=4, expert_activation=nn.LeakyReLU, gating_activation=nn.ReLU, use_expert_bias=True, use_gating_bias=True):
        super(MulMoE, self).__init__()
        self.units = units
        self.n_experts = n_experts
        self.expert_input_dims = expert_input_dims
        self.feature_dim = feature_dim
        self.n_heads = n_heads

        self.expert_activation = expert_activation()
        self.gating_activation = gating_activation()
        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        # Expert weights and biases
        self.expert_kernels = nn.ModuleList()
        self.expert_biases = nn.ParameterList()
        for i, expert_dim in enumerate(self.expert_input_dims):
            expert_kernel = torch.nn.Sequential(
                nn.Linear(expert_dim, self.units, bias=self.use_expert_bias),
                nn.BatchNorm1d(self.units),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.expert_kernels.append(expert_kernel)
            if self.use_expert_bias:
                expert_bias = nn.Parameter(torch.zeros(self.units))
                self.expert_biases.append(expert_bias)

        # Multi-head Gating weights and biases
        self.gating_kernels = nn.ModuleList([nn.Linear(self.feature_dim, self.n_experts, bias=self.use_gating_bias) for _ in range(n_heads)])
        if self.use_gating_bias:
            self.gating_biases = nn.ParameterList([nn.Parameter(torch.zeros(self.n_experts)) for _ in range(n_heads)])

        self.global_weights = nn.Parameter(torch.ones(self.n_experts), requires_grad=True)

    def forward(self, feature_input, inputs):
        expert_outputs = []
        expert_inputs = torch.split(inputs, self.expert_input_dims, dim=-1)

        # Compute expert outputs
        for i, expert_input in enumerate(expert_inputs):
            expert_output = self.expert_kernels[i](expert_input)
            if self.use_expert_bias:
                expert_output += self.expert_biases[i]
            if self.expert_activation is not None:
                expert_output = self.expert_activation(expert_output)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch_size, units, n_experts]

        # Apply multi-head gating with sparse top-k experts
        multi_head_outputs = []
        for i in range(self.n_heads):
            gating_outputs = self.gating_kernels[i](feature_input)  # [batch_size, n_experts]
            if self.use_gating_bias:
                gating_outputs += self.gating_biases[i]
            if self.gating_activation is not None:
                gating_outputs = self.gating_activation(gating_outputs)

            normalized_global_weights = torch.softmax(self.global_weights / 0.01, dim=0)
            weighted_gating_outputs = gating_outputs * normalized_global_weights
            top_k_values, top_k_indices = torch.topk(weighted_gating_outputs, k=4, dim=-1)  # [batch_size, 4] $4
            normalized_top_k_values = torch.softmax(top_k_values / 0.01, dim=-1)  # [batch_size, 4]
            selected_experts = torch.gather(expert_outputs, -1, top_k_indices.unsqueeze(1).expand(-1, self.units, -1))  # [batch_size, units, 4]
            weighted_experts = selected_experts * normalized_top_k_values.unsqueeze(1)  # [batch_size, units, 4]

            head_output = weighted_experts.sum(dim=-1)  # [batch_size, units]
            multi_head_outputs.append(head_output)

        output = torch.cat(multi_head_outputs, dim=-1)  # [batch_size, units * n_heads]

        return output, expert_outputs


class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=954,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        
        self.drug_dim = [768, 512, 512, 256, 768, 256, 768, 512]
        self.cell_dim = [512, 512, 512, 256, 1536, 3072, 1120, 919]

        # self.cell_moe = MulMoE(units=128, n_experts=len(self.cell_dim), expert_input_dims=self.cell_dim, feature_dim=num_features_xt, n_heads=8)
        self.cell_moe = MulMoE(units=1024, n_experts=len(self.cell_dim), expert_input_dims=self.cell_dim, feature_dim=num_features_xt, n_heads=8)
        self.global_weights = self.cell_moe.global_weights

        self.moe_nn = nn.Sequential(
            nn.Linear(1024*8, 2048), 
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, 1024)
        self.drug1_conv2 = GCNConv(1024, 512)
        self.drug1_conv3 = GCNConv(512, num_features_xd * 2)
        # self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        # SMILES2 graph branch
        self.drug2_conv1 = GCNConv(num_features_xd, 1024)
        self.drug2_conv2 = GCNConv(1024, 512)
        self.drug2_conv3 = GCNConv(512, num_features_xd * 2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)


        # DL cell featrues
        # self.reduction = nn.Sequential(
        #     nn.Linear(8439, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim)
        # )

        self.drug_pre = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data1, data2):

        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        cell_all = torch.cat((data1.cellplm, data1.celllm, data1.scgpt, data1.geneformer, data1.genept, data1.scf, data1.scmulan, data1.scbert), dim=1)

        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g2(x1))
        x1 = self.dropout(x1)


        # deal drug2
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug1_fc_g2(x2))
        x2 = self.dropout(x2)

        cell_all_ = F.normalize(cell_all, 2, 1)
        cell_t, expert_outputs = self.cell_moe(cell, cell_all_)
        cell_t = self.moe_nn(cell_t)
        info_loss = infonce_loss_cosine(expert_outputs)

        # concat
        xc = torch.cat((x1, x2, cell_t), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out, info_loss
