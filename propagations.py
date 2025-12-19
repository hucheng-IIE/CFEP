import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import math
# Graph Propagation models
from dgl.nn import GraphConv
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from utils import *
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class CompGCN_dg_AEPE_sem(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.W_mean = nn.Linear(self.node_in_feat, 1)
        self.W_sum = nn.Linear(self.node_out_feat, 1)

        if self.bias:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, g, reverse=False, top_k=3):
        def custom_message_func(edges):
            return {'m': edges.src['h']}

        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                # 自环连接处理
                h_loop = self.msg_loop_linear(nodes.data['h'])
                h = h + h_loop
            if self.dropout:
                h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def custom_agg_mean(nodes):
            neighbors_feature = nodes.mailbox['m']
            if neighbors_feature.shape[1] <= top_k:
                feature = torch.mean(neighbors_feature, dim=1)
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_mean(neighbors_weights)
                _, select_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                select_feature = torch.gather(neighbors_feature, 1, select_indices.expand(-1, -1, neighbors_feature.shape[2]))
                feature = torch.mean(select_feature, dim=1)
            return {'h_o_r': feature}

        def custom_agg_sum(nodes):
            neighbors_feature = nodes.mailbox['m']
            if neighbors_feature.shape[1] <= top_k:
                feature = torch.sum(neighbors_feature, dim=1)
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_sum(neighbors_weights)
                _, select_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                select_feature = torch.gather(neighbors_feature, 1, select_indices.expand(-1, -1, neighbors_feature.shape[2]))
                feature = torch.sum(select_feature, dim=1)
            return {'h': feature}

        # ---------- 主处理流程 ----------
        # 第一轮消息传递
        g.update_all(custom_message_func, custom_agg_mean)
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        
        # 存储中间特征
        g.ndata['h_s_r_o'] = h_o_r
        
        # 第二轮聚合
        g.update_all(fn.copy_u('h_s_r_o', 'm'), custom_agg_sum, apply_func)

        return g.ndata['h']

# CompGCN based on direct graphs. We do not have inversed edges
class CompGCN_dg_AEPE(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.W_mean = nn.Linear(self.node_in_feat,1)
        self.W_sum = nn.Linear(self.node_out_feat,1)

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)     
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, g, reverse=False, top_k=3): 
        
        def get_neighbor_counts(g):
            neighbor_counts = g.out_degrees().tolist() 
            for node, count in enumerate(neighbor_counts):
                print(f"Node {node} has {count} neighbors.")
        
        def custom_message_func(edges):
            message = edges.src['h'] * edges.data['text_h']
            return {'m': message}

        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['text_h'])
            return {'text_h': e_h}

        def custom_agg_mean(nodes):
            neighbors_feature = nodes.mailbox['m'] # (n_nodes, n_neighbors, feature_dim)
            if neighbors_feature.shape[1] <=3:  #neighbor num
                feature = torch.mean(neighbors_feature, dim=1)
                return {'h_o_r': feature}
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_mean(neighbors_weights)
                #select k neighors
                _, select_neighbors_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                #select message
                select_neighbors_feature = torch.gather(neighbors_feature, dim=1, index=select_neighbors_indices.expand(-1, -1, neighbors_feature.shape[2]))
                #agg
                meg = torch.mean(select_neighbors_feature, dim=1)  # (n_nodes, feature_dim)
                return {'h_o_r': meg}

        def custom_agg_sum(nodes):
            neighbors_feature = nodes.mailbox['m'] # (n_nodes, n_neighbors, feature_dim)
            if neighbors_feature.shape[1] <=3:  #neighbor num
                feature = torch.sum(neighbors_feature, dim=1)
                return {'h': feature}
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_sum(neighbors_weights)
                #select k neighors
                _, select_neighbors_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                #select message
                select_neighbors_feature = torch.gather(neighbors_feature, dim=1, index=select_neighbors_indices.expand(-1, -1, neighbors_feature.shape[2]))
                #agg
                meg = torch.sum(select_neighbors_feature, dim=1)  # (n_nodes, feature_dim)
                return {'h': meg}

        #get_neighbor_counts(g)
        #g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.mean('m', 'h_o_r'))
        g.update_all(custom_message_func, custom_agg_mean) 
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        g.ndata['h_s_r_o'] = h_o_r
        #g.update_all(fn.copy_u(u='h_s_r_o', out='m'), fn.sum(msg='m', out='h'),apply_func)
        g.update_all(fn.copy_u(u='h_s_r_o', out='m'), custom_agg_sum, apply_func)
        g.apply_edges(apply_edge) 

# Graph Propagation models

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)
        return {'h' : h}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.0):
        super().__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, feature):
        def gcn_msg(edge):
            msg = edge.src['h'] * edge.data['w'].float()
            return {'m': msg}
 
        #feature = g.ndata['h']
        if self.dropout:
            feature = self.dropout(feature)

        g.ndata['h'] = feature
        g.update_all(gcn_msg, fn.sum(msg='m', out='h'))
        g.apply_nodes(func=self.apply_mod)
        return g.ndata['h']


# # CompGCN based on direct graphs. We do not have inversed edges
class CompGCN_dg_mtg(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, sentence_size, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)
        self.text_linear = nn.Linear(sentence_size, node_in_feat)
        self.msg_inv_linear = nn.Linear(node_in_feat*2, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)
        self.rel_linear = nn.Linear(rel_in_feat*2, rel_out_feat, bias=bias) # w@e_r

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, g, reverse=False):

        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = torch.cat([edges.data['e_h'], edges.data['s_h_']], dim=1)
            e_h = self.rel_linear(e_h)
            return {'e_h': e_h}
 
        #print ("s_h:",g.edata['s_h'].shape)
        g.edata['s_h_'] = self.text_linear(g.edata['s_h'])
        g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.mean('m', 'h_o_r'))
        g.update_all(fn.copy_e('s_h_', 'm'), fn.mean('m', 'h_o_s'))

        g.ndata['h_o_cat'] = torch.cat([g.ndata['h_o_s'], g.ndata['h_o_r']], dim=1)
        h_o_r = self.msg_inv_linear(g.ndata['h_o_cat'])
        g.ndata['h_s_r_o'] = h_o_r
        g.update_all(fn.copy_u('h_s_r_o', 'm'), fn.sum(msg='m', out='h'), apply_func)
        g.apply_edges(apply_edge)

# CompGCN based on direct graphs. We do not have inversed edges
class CompGCN_dg_glean(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)     
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, g, reverse=False):  
      
        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['e_h'])
            return {'e_h': e_h}

        g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.mean('m', 'h_o_r')) 
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        g.ndata['h_s_r_o'] = h_o_r 
        g.update_all(fn.copy_u(u='h_s_r_o', out='m'), fn.sum(msg='m', out='h'),apply_func)
        g.apply_edges(apply_edge)
#GCN
class GCN_PECF(nn.Module):
    def __init__(self, node_in_feats, node_n_hidden, node_out_feats,rel_in_feat,rel_out_feat, n_layers, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat) # w@e_r
        if n_layers < 2:
            self.layers.append(GraphConv(node_in_feats, node_out_feats, activation=activation))
        else:
            self.layers.append(GraphConv(node_in_feats, node_n_hidden, activation=activation))
            for i in range(n_layers - 2):
                self.layers.append(GraphConv(node_n_hidden, node_n_hidden, activation=activation))
            self.layers.append(GraphConv(node_n_hidden, node_out_feats, activation=activation)) # activation or None

    def forward(self, g): # no reverse
        g = dgl.add_self_loop(g)
        def apply_func(nodes):
            h = nodes.data['h']
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['text_h'])
            return {'text_h': e_h}
        
        for layer in self.layers:
            g.ndata['h'] = layer(g, g.ndata['h'])
        g.apply_edges(apply_edge)
        g.apply_nodes(apply_func)
    
# CompGCN based on direct graphs. We do not have inversed edges, PECF
class CompGCN_dg_PECF(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)     
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, g, reverse=False):  
      
        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['text_h'])
            return {'text_h': e_h}

        g.update_all(fn.v_mul_e('h', 'text_h', 'm'), fn.mean('m', 'h_o_r')) 
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        g.ndata['h_s_r_o'] = h_o_r 
        g.update_all(fn.copy_u(u='h_s_r_o', out='m'), fn.sum(msg='m', out='h'),apply_func)
        g.apply_edges(apply_edge)

class tRGCN_dg(nn.Module):
    def __init__(self, node_in_feat, hid_dim,node_out_feat,rel_in_feat,rel_out_feat,
                 n_layers,num_rels,regularizer="basis",num_bases=None,
                 use_bias=True,activation=F.relu, use_self_loop=True,
                 layer_norm=False,low_mem=False, dropout=0.0):
        super().__init__()
        self.in_dim = node_in_feat
        self.out_dim = node_out_feat
        self.hid_dim = hid_dim
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.n_layers = n_layers
        self.bias = use_bias
        self.activation = activation
        self.self_loop = use_self_loop
        self.msg_loop_linear = nn.Linear(node_out_feat, node_out_feat, bias=use_bias)     

        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.layer_norm = layer_norm
        self.low_mem = low_mem
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=use_bias) # w@e_r
        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        assert self.n_layers >= 1, self.n_layers
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm,
            ))
        else:
            # i2h
            self.layers.append(RelGraphConv(
                self.in_dim, self.in_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm
            ))
            # h2h
            for i in range(1, self.n_layers - 1):
                self.layers.append(RelGraphConv(
                    self.in_dim, self.in_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                    activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                    layer_norm=self.layer_norm
                ))
            # h2o
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm
            ))
        assert self.n_layers == len(self.layers), (self.n_layers, len(self.layers))

    def forward(self, g): 
        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}
        
        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['e_h'])
            return {'e_h': e_h}
        
        edge_norm = node_norm_to_edge_norm(g)
        node_embs = g.ndata['h']
        edge_etype = g.edata['rel_type'].long()

        for layer in self.layers:
                node_embs = layer(g, node_embs, edge_etype, edge_norm)

        g.ndata['h'] = node_embs
        g.apply_nodes(apply_func)
        g.apply_edges(apply_edge)

# R-GCN based on direct graphs. We do not have inversed edges
class RGCN_dg(nn.Module):
    def __init__(self, node_in_feat, hid_dim,node_out_feat,rel_in_feat,rel_out_feat,
                 n_layers,num_rels,regularizer="basis",num_bases=None,
                 use_bias=True,activation=F.relu, use_self_loop=True,
                 layer_norm=False,low_mem=False, dropout=0.0,text_emb_dim=None):
        super().__init__()
        self.in_dim = node_in_feat
        self.out_dim = node_out_feat
        self.hid_dim = hid_dim
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.n_layers = n_layers
        self.bias = use_bias
        self.activation = activation
        self.self_loop = use_self_loop
        self.msg_loop_linear = nn.Linear(node_out_feat, node_out_feat, bias=use_bias)     

        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.layer_norm = layer_norm
        self.low_mem = low_mem
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=use_bias) # w@e_r
        #self.text_linear = nn.Linear(text_emb_dim,node_in_feat)
        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        #self.rel_linear = nn.Linear(rel_in_feat*3, rel_out_feat, bias=use_bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        assert self.n_layers >= 1, self.n_layers
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm,
            ))
        else:
            # i2h
            self.layers.append(RelGraphConv(
                self.in_dim, self.in_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm
            ))
            # h2h
            for i in range(1, self.n_layers - 1):
                self.layers.append(RelGraphConv(
                    self.in_dim, self.in_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                    activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                    layer_norm=self.layer_norm
                ))
            # h2o
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.bias,
                activation=self.activation, self_loop=self.self_loop, dropout=dropout,
                layer_norm=self.layer_norm
            ))
        assert self.n_layers == len(self.layers), (self.n_layers, len(self.layers))

    def forward(self, g): 
        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}
        
        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['text_h'])
            return {'text_h': e_h}
        
        edge_norm = node_norm_to_edge_norm(g)
        node_embs = g.ndata['h']
        edge_etype = g.edata['rel_type'].long()

        for layer in self.layers:
                node_embs = layer(g, node_embs, edge_etype, edge_norm)

        g.ndata['h'] = node_embs
        g.apply_nodes(apply_func)
        g.apply_edges(apply_edge)

class MaskLinear(Module):
    def __init__(self, in_features, out_features=1, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.Tensor(in_features)) 
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, idx): # idx is a list
        mask = torch.zeros(self.in_features, self.out_features).cuda()
        mask[idx] = x.squeeze()
        output = torch.matmul(self.weight, mask)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' => ' \
               + str(self.out_features) + ')'

class TemporalEncoding(Module):
    def __init__(self, in_features, bias=True): 
        super(TemporalEncoding, self).__init__()
        out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, out_o))  
        self.weight_c = Parameter(torch.Tensor(in_features, out_c))
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_c.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features)) 
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_o, h_c):  
        trans_ho = torch.mm(h_o, self.weight_o)  
        trans_hc = torch.mm(h_c, self.weight_c)  
        output =torch.tanh( (torch.cat((trans_ho, trans_hc), dim=1))) # dim=1

        if self.bias is not None:
            return output + self.bias
        else:
            return output
