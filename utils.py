import numpy as np
import os
from math import log
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

import dgl
import json
import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss,balanced_accuracy_score,accuracy_score
from sklearn.metrics import jaccard_score
from event_data_processing import *
import time
from tqdm import tqdm

def compute_prediction_set(logits, eta):
    # logits: (N, K)
    probs = torch.softmax(logits, dim=-1)
    s_k = 1.0 - probs
    # prediction set: classes where s_k <= eta
    mask = (s_k <= eta.unsqueeze(0).unsqueeze(1))  # (N, K) bool
    return mask

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)
  
class MergeLayer_tgn(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    #x = torch.cat([x1, x2], dim=1)
    x = self.fc1(x)
    h = self.act(self.fc2(x))
    return h


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

def get_sentence_embeddings(graph_dict, source_cache, edge_times, device):
    # n = len(story_ids)
    # sentence_embeddings = torch.nn.Parameter(torch.zeros(n, embedding_size), requires_grad=True).to(device)
    # nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    n = len(source_cache)
    edge_times = edge_times.tolist()
    time = list(set(edge_times))
    g_list = graph_dict[time[0]]
    sentence_embeddings = g_list.edata['text_emb'].to(device)
    if len(sentence_embeddings) != n:
       print("!!")
       print(time)
       sentence_embeddings = torch.nn.Parameter(torch.zeros(n, 768), requires_grad=True).to(device)
       nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    return sentence_embeddings

def node_norm_to_edge_norm(G):
    G = G.local_var()
    G.apply_edges(lambda edges: {'norm': edges.dst['norm']})

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_y_data(data, target_rels):
    """ (# of s-related triples) / (total # of triples) """

    time_l = list(set(data[:,-1]))
    time_l = sorted(time_l,reverse=False)
    y_data = []
    for cur_t in time_l:
        has_Y = 0
        triples = get_data_with_t(data, cur_t)
        r_arr = triples[:,1]
        for rel in r_arr:
            if rel in target_rels:
                has_Y = 1
                break
        y_data.append(has_Y)
    # y_data = torch.Tensor(y_data)
    return y_data

def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None
 
 
def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    #get times
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)
 
'''
Customized collate function for Pytorch data loader
'''
def collate_2(batch):
    batch_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    return [batch_data, y_data]
 
def collate_4(batch):
    batch_data = [item[0] for item in batch]
    s_prob = [item[1] for item in batch]
    r_prob = [item[2] for item in batch]
    o_prob = [item[3] for item in batch]
    return [batch_data, s_prob, r_prob, o_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

 
'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]  
    idx_zero = idx[num_non_zero-1:]  
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r 
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids) 
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None
    
    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero
 
def get_neighbor_finder(args, inPath, fileName, fileName2, fileName3, fileName4, fileName5, uniform, num_node, num_rels):
    adj_list = [[] for _ in range(num_node)]
    rel_info = [[] for _ in range(num_rels)]
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            rel_info[rel].append((head, tail, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            rel_info[rel].append((head, tail, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            rel_info[rel].append((head, tail, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName4), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            rel_info[rel].append((head, tail, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName5), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            rel_info[rel].append((head, tail, time))
            # adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(args, rel_info, adj_list, uniform=uniform)

class NeighborFinder:
    def __init__(self, args, rel_info, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.rel_to_timestamps = []
        self.rel_to_nodes = []
        #self.node_to_rel = [[] for _ in len(node_to_neighbors)]

        for src, neighbors in enumerate(adj_list):
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))   #node -> related rel
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))

        self.timestamps_to_rel = defaultdict(set)
        for rel, info in enumerate(rel_info):
            #self.rel_to_timestamps.append(np.array([x[2] for x in info]))
            nodes = set()
            for x in info:
                nodes.add(x[0])
                nodes.add(x[1])
                self.timestamps_to_rel[x[2]].add(rel)
            self.rel_to_nodes.append(nodes)             #set
        
        self.cache_dir = args.dp + args.dataset
        self.cache_file = os.path.join(self.cache_dir, "rel_shared_neighbors.pkl")
        jacc_martix = self.compute_jaccard_matrix()
        self.rel_shared_map = self.build_rel_shared_neighbors(jacc_martix)

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        
        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                        timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
        if len(source_neighbors) > 0 and n_neighbors > 0:
            if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                neighbors[i, :] = source_neighbors[sampled_idx]
                edge_times[i, :] = source_edge_times[sampled_idx]
                edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                # re-sort based on time
                pos = edge_times[i, :].argsort()
                neighbors[i, :] = neighbors[i, :][pos]
                edge_times[i, :] = edge_times[i, :][pos]
                edge_idxs[i, :] = edge_idxs[i, :][pos]
            else:
                # Take most recent interactions
                source_edge_times = source_edge_times[-n_neighbors:]
                source_neighbors = source_neighbors[-n_neighbors:]
                source_edge_idxs = source_edge_idxs[-n_neighbors:]

                assert (len(source_neighbors) <= n_neighbors)
                assert (len(source_edge_times) <= n_neighbors)
                assert (len(source_edge_idxs) <= n_neighbors)

                neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times

    def compute_jaccard_matrix(self):

        n_rel = len(self.rel_to_nodes)
        sim_matrix = np.zeros((n_rel, n_rel), dtype=float)
        
        for i in range(n_rel):
            for j in range(i, n_rel):
                if i == j:
                    sim = 1.0
                else:
                    inter = self.rel_to_nodes[i] & self.rel_to_nodes[j]
                    union = self.rel_to_nodes[i] | self.rel_to_nodes[j]
                    sim = len(inter) / len(union) if len(union) > 0 else 0.0
                sim_matrix[i, j] = sim_matrix[j, i] = sim

        return sim_matrix

    def build_rel_shared_neighbors(self, jacc_martix, threshold = 0.3):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                rel_shared_map = pickle.load(f)
            return rel_shared_map
        
        n = jacc_martix.shape[0]
        rel_shared_map = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if jacc_martix[i,j] > threshold:
                    rel_shared_map[i].append(j)         #get similty event

        # with open(self.cache_file, "wb") as f:
        #     pickle.dump(rel_shared_map, f)
        # print(f"[Cache] Saved computed neighbors to {self.cache_file}")

        return rel_shared_map

    def build_neighbors_struct(self, t):
        """
        给定时间 t，返回每条边在时刻 t 的结构邻居边。
        共享节点 + 同一时刻。
        """
        t_val = t.detach().cpu().item() if torch.is_tensor(t) else t

        neighbors_struct = []
        for rel, related_rel_set in enumerate(self.rel_shared_map):
            related_rel = list(set(related_rel_set) & set(self.timestamps_to_rel[t_val]))
            neighbors_struct.append(related_rel)
       
        return neighbors_struct

    def build_neighbors_temporal(self, t, w):
        """
        给定时间窗口 w，返回每条边在时间区间 [t - w, t) 内的时间邻居边。
        共享节点 + 时间条件。
        """
        t_val = t.detach().cpu().item() if torch.is_tensor(t) else t
        t_start = max(0, t_val - w)
        
        neighbors_temporal = []
        for rel, related_rel_set in enumerate(self.rel_shared_map):
            merged_rels = set()
            for t in range(t_start, t_val + 1):
                merged_rels |= self.timestamps_to_rel[t]

            related_rel = list(set(related_rel_set) & merged_rels)
            neighbors_temporal.append(related_rel)

        return neighbors_temporal

def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    
def gcl_loss_def(z1, z2, temperature=0.1):
    batch_size, num_rels, dim = z1.size()
    
    # g_z1 = self.gcl_mlp(z1)  # (B, N, D)
    # g_z2 = self.gcl_mlp(z2)
    
    g_z1 = z1
    g_z2 = z2

    g_z1_flat = g_z1.view(-1, dim)  # (B*N, D)
    g_z2_flat = g_z2.view(-1, dim)
    
    sim_matrix = F.cosine_similarity(g_z1_flat.unsqueeze(1), g_z2_flat.unsqueeze(0), dim=2)  # (B*N, B*N)
    sim_matrix = sim_matrix / temperature
    
    pos_mask = torch.eye(batch_size * num_rels, dtype=torch.bool, device=z1.device)
    
    logits = torch.exp(sim_matrix)
    logits_pos = logits[pos_mask]
    logits_neg = logits[~pos_mask].view(batch_size * num_rels, -1).sum(dim=1)
    loss = -torch.mean(torch.log(logits_pos / (logits_pos + logits_neg)))
    
    return loss

'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o
 
'''
Empirical distribution(total # of triples)
'''
def get_true_distributions(path, data, num_nodes, num_rels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    #file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)
    file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)      #cp dataset
    if not os.path.exists(file_path):
        print('build true distributions...',dataset,set_name)
        time_l = list(set(data[:,-1]))
        time_l = sorted(time_l,reverse=False)
        true_prob_s = None
        true_prob_o = None
        true_prob_r = None
        for cur_t in time_l:
            triples = get_data_with_t(data, cur_t)
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
            true_r = np.zeros(num_rels)
            s_arr = triples[:,0]
            o_arr = triples[:,2]
            r_arr = triples[:,1]
            for s in s_arr:
                true_s[s] += 1
            for o in o_arr:
                true_o[o] += 1
            for r in r_arr:
                true_r[r] += 1
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            true_r = true_r / np.sum(true_r)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
                true_prob_r = true_r.reshape(1, num_rels)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, num_rels)), axis=0)
             
        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_s,true_prob_r,true_prob_o], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_s, true_prob_r, true_prob_o] = pickle.load(f)
    true_prob_s = torch.from_numpy(true_prob_s)
    true_prob_r = torch.from_numpy(true_prob_r)
    true_prob_o = torch.from_numpy(true_prob_o)
    return true_prob_s, true_prob_r, true_prob_o 

'''
Evaluation metrics
'''
# Label based
 
def print_eval_metrics_Bin(y_true, y_predict, prt=True):
    y_true = [item for sublist in y_true for item in sublist]
    y_predict = [item for sublist in y_predict for item in sublist]
    y_predict = [x>0.5 for x in y_predict]
    
    recall = recall_score(y_true, y_predict, average='binary')
    f1 = f1_score(y_true, y_predict, average='binary')
    beta=2
    f2 = fbeta_score(y_true, y_predict, average='binary', beta=beta)
    hloss = hamming_loss(y_true, y_predict)
    bacc = balanced_accuracy_score(y_true,y_predict)
    acc = accuracy_score(y_true,y_predict)
    if prt:
        print("Rec  weighted: {:.4f}".format(recall))
        print("F1  weighted: {:.4f}".format(f1))
        print("F{}  weighted: {:.4f}".format(beta,f2))
        print("hamming loss: {:.4f}".format(hloss))
        print("bacc: {:.4f}".format(bacc))
        print("acc: {:.4f}".format(acc))
    return hloss, recall, f1, f2, bacc, acc

def print_eval_metrics(true_rank_l, prob_rank_l, prt=True): #?
    m = MultiLabelBinarizer().fit(true_rank_l)
    m_actual = m.transform(true_rank_l)
    m_predicted = m.transform(prob_rank_l)
    #weighted,micro,macro
    p = precision_score(m_actual, m_predicted, average='micro')
    recall = recall_score(m_actual, m_predicted, average='micro')
    f1 = f1_score(m_actual, m_predicted, average='micro')
    beta=2
    f2 = fbeta_score(m_actual, m_predicted, average='micro', beta=beta)
    hloss = hamming_loss(m_actual, m_predicted)
    #bacc = balanced_accuracy_score(m_actual, m_predicted)
    accuracy_per_l = (m_actual == m_predicted).mean(axis=0)
    acc = accuracy_per_l.mean()
    #acc = accuracy_score(m_actual, m_predicted)
    if prt:
        print("Pre weighted:{:.4f}".format(p))
        print("Rec  weighted: {:.4f}".format(recall))
        print("F1  weighted: {:.4f}".format(f1))
        print("F{}  weighted: {:.4f}".format(beta,f2))
        print("hamming loss: {:.4f}".format(hloss))
        # print("bacc: {:.4f}".format(bacc))
        print("acc: {:.4f}".format(acc))
    return hloss, p, recall, f1, f2, acc

def print_hit_eval_metrics(total_ranks):
    total_ranks += 1
    mrr = np.mean(1.0 / total_ranks) 
    mr = np.mean(total_ranks)
    hits = []
    for hit in [1, 3, 10]: # , 20, 30
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("Hits @ {}: {:.4f}".format(hit, avg_count))
    # print("MRR: {:.4f} | MR: {:.4f}".format(mrr,mr))
    return hits, mrr, mr

def adaptive_node_drop(g, node_embeds, drop_prob=0.2):
    features = node_embeds[g.ndata['id']]
    node_importance = torch.norm(features, p=2, dim=1)
    threshold = torch.quantile(node_importance, drop_prob)
    
    keep_nodes = [i for i, score in enumerate(node_importance) if score > threshold]
    sg = dgl.node_subgraph(g, keep_nodes)
    return sg

def adaptive_edge_drop(g, drop_prob=0.2):
    num_edges = g.num_edges()
    keep_edges = random.sample(range(num_edges), int(num_edges * (1 - drop_prob)))
    sg = dgl.edge_subgraph(g, keep_edges)
    return sg

def generate_adv_graph(orig_graph, model, delta_A=5, delta_X=10, lr=0.01, epochs=50):
    """
    生成包含结构和特征扰动的对抗图
    返回: adv_graph (dgl.DGLGraph)
    """
    A = orig_graph.adjacency_matrix().to_dense().clone().float()
    X = orig_graph.ndata['h'].clone().float()           #event embedding
    device = A.device
    
    A.requires_grad_(True)
    X.requires_grad_(True)
    
    optimizer = torch.optim.Adam([A, X], lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        
        src, dst = A.nonzero(as_tuple=True)
        tmp_g = dgl.graph((src, dst), num_nodes=orig_graph.num_nodes()).to(device)
        tmp_g.ndata['feat'] = X
        
        Z = model(tmp_g, tmp_g.ndata['feat'])
        
        # 计算对比损失
        loss = contrastive_loss(Z)
        
        # 反向传播
        loss.backward()
        GA, GX = A.grad.data, X.grad.data
        
        # 应用结构扰动
        with torch.no_grad():
            # 边添加：选择正梯度最大的非现有边
            pos_mask = (GA > 0) & (A < 0.5)
            add_indices = torch.topk(GA[pos_mask], delta_A).indices
            A_flat = A.flatten()
            A_flat[pos_mask.flatten()][add_indices] = 1
            
            # 边删除：选择负梯度最大的现有边
            neg_mask = (GA < 0) & (A > 0.5)
            del_indices = torch.topk(-GA[neg_mask], delta_A).indices
            A_flat[neg_mask.flatten()][del_indices] = 0
            A.data = A_flat.view_as(A)
        
        # 应用特征扰动
        with torch.no_grad():
            # 生成特征掩码（掩码负梯度最大的特征维度）
            neg_grad_mask = (GX < 0)
            grad_abs = torch.abs(GX)
            topk_values, topk_indices = torch.topk(grad_abs[neg_grad_mask], delta_X)
            M = torch.ones_like(X)
            M.view(-1)[topk_indices] = 0
            X.data = X * M
    
    # 构建最终对抗图
    src, dst = A.nonzero(as_tuple=True)
    adv_graph = dgl.graph((src, dst), num_nodes=orig_graph.num_nodes())
    adv_graph.ndata['feat'] = X.detach()
    return adv_graph

def add_noise_to_graphs_gca(g_list,node_embeds):
    noisy_graphs = []
    for g in g_list:
        noisy_g = adaptive_edge_drop(g)
        noisy_g = adaptive_node_drop(noisy_g,node_embeds)
        noisy_graphs.append(noisy_g)
    return noisy_graphs

def add_noise_to_graphs_graphcl(g_list):
    noisy_graphs = []
    for g in g_list:
        noisy_g = augment_graph(g)
        noisy_graphs.append(noisy_g)
    return noisy_graphs

def get_noisy_graph_with_features(g, drop_prob=0.3, add_prob=0.1):
    num_edges = g.num_edges()
    mask = torch.rand(num_edges) > drop_prob
    edge_ids = torch.arange(num_edges)[mask]

    # Clone the original graph
    noisy_g = g.clone()
    noisy_g.remove_edges(torch.arange(num_edges)[~mask])  # Remove edges not in mask
    noisy_g.edata['text_emb'] = g.edata['text_emb'][mask]
    
    return noisy_g

def augment_graph(g, drop_node_prob=0.2):
    num_nodes = g.num_nodes()
    keep_nodes = [i for i in range(num_nodes) if random.random() > drop_node_prob]
    sg = dgl.node_subgraph(g, keep_nodes)
    return sg

def gcl_loss(dp, dg, tau=1.0):
    
    # sim = torch.matmul(dp, dg.transpose(1, 2)) / tau
    # loss = -torch.mean(torch.log(torch.exp(sim) / torch.sum(torch.exp(sim), dim=2, keepdim=True)))

    sim = torch.matmul(dp, dg.transpose(1, 2)) / tau
    log_probs = F.log_softmax(sim, dim=2)
    loss = -torch.mean(log_probs)

    return loss

def shuffle_windowed_edge_temporal(windowed_g_list):
    """
    时间窗口级别的边特征时序打乱
    """
    windowed_shuffle_g_list = []
    
    for window in windowed_g_list:
    
        T = len(window)
        device = window[0].edata['text_emb'].device  # 统一设备
        
        # ==== 步骤1: 收集所有时间步的边及特征 ====
        edge_dict = {}  # key: (src, dst), value: List[(time_step, feature)]
        for t, g in enumerate(window):
            src, dst = g.edges()
            src = src.cpu().numpy()
            dst = dst.cpu().numpy()
            feats = g.edata['text_emb']
            
            # 记录当前时间步的边特征
            for i in range(g.num_edges()):
                edge_key = (int(src[i]), int(dst[i]))
                if edge_key not in edge_dict:
                    edge_dict[edge_key] = []
                edge_dict[edge_key].append( (t, feats[i]) )

        # ==== 步骤2: 对每个边的时序特征独立打乱 ====
        shuffled_edge_dict = {}
        for edge_key, feat_list in edge_dict.items():
            # 提取该边在所有时间步的特征（可能不连续）
            times = [t for t, _ in feat_list]
            features = [f.clone() for _, f in feat_list]
            
            # 打乱特征顺序
            shuffle_idx = torch.randperm(len(features))
            shuffled_features = [features[i] for i in shuffle_idx]
            
            # 记录打乱后的特征
            for i, t in enumerate(times):
                if edge_key not in shuffled_edge_dict:
                    shuffled_edge_dict[edge_key] = {}
                shuffled_edge_dict[edge_key][t] = shuffled_features[i]

        # ==== 步骤3: 重建每个时间步的图 ====
        new_window = []
        for t in range(T):
            orig_g = window[t]
            src, dst = orig_g.edges()
            
            # 获取原始边集合
            edge_keys = list(zip(src.cpu().numpy().tolist(), 
                               dst.cpu().numpy().tolist()))
            
            # 收集打乱后的特征（保持原边顺序）
            shuffled_feats = []
            for edge_key in edge_keys:
                if edge_key in shuffled_edge_dict and t in shuffled_edge_dict[edge_key]:
                    shuffled_feats.append(shuffled_edge_dict[edge_key][t])
                else:
                    # 若该时间步无此边（理论上不会出现，因从原始数据生成）
                    shuffled_feats.append(orig_g.edata['text_emb'][edge_keys.index(edge_key)])
            
            # 构建新图
            new_g = orig_g.clone()
            #new_g = dgl.graph((src, dst), num_nodes=orig_g.num_nodes()).to(device)
            new_g.edata['text_emb'] = torch.stack(shuffled_feats, dim=0)
            new_window.append(new_g)
        
        windowed_shuffle_g_list.append(new_window)
    
    return windowed_shuffle_g_list

def comp_deg_norm(g):
    """计算归一化系数"""
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def build_knn_graph_batch(feats, k=5, device='cpu'):
    """批量构建KNN图并注入节点特征与归一化系数"""
    # knn_graphs = []
    
    # # features_batch形状: [batch_size, num_nodes, dim]
    # for i in range(features_batch.size(0)):
    #     feats = features_batch[i]  # [num_nodes, dim]
        
    # 计算余弦相似度
    norm_feats = F.normalize(feats, p=2, dim=1)
    sim_matrix = torch.mm(norm_feats, norm_feats.t())  # [num_nodes, num_nodes]
    
    # 获取top-k邻居（排除自身）
    topk_values, topk_indices = torch.topk(sim_matrix, k+1, dim=1)
    indices = topk_indices[:, 1:]  # [num_nodes, k]
    
    # 构建边列表
    src = torch.repeat_interleave(torch.arange(feats.size(0), device=device), k)
    dst = indices.reshape(-1).to(device)
    
    # 创建无向图
    g = dgl.graph(
        (torch.cat([src, dst]), torch.cat([dst, src])),
        num_nodes=feats.size(0),
        device=device
    )
    
    # 注入节点特征
    g.ndata['h'] = feats.to(device)  # 关键修改：赋予节点特征
    # 计算并注入归一化系数
    norm = comp_deg_norm(g).to(device)
    g.ndata['norm'] = norm.view(-1, 1)
    
    return g

def generate_knn_for_windows(num_rels,node_in_feat,windowed_g_list, model, t_list, k=5):
    """
    生成时间窗口的KNN图列表
    输入:
        windowed_g_list: List[List[DGLGraph]] 时间窗口列表
        model: 返回边嵌入的模型 (需实现get_edge_emb方法)
        k: KNN参数
    输出:
        windowed_KNN_g_list: List[List[DGLGraph]] 对应的KNN图列表
    """
    device = next(model.parameters()).device
    
    window_knn_graphs = []
    # model.eval()
    #  # ---------- 冻结模型参数 ----------
    # for p in model.parameters():
    #     p.requires_grad_(False)

    for window in windowed_g_list:
        knn_graphs = []
        for g in window:
            node_emb = get_rel_embedding(num_rels,node_in_feat*24,g,g.edata['text_emb'])
            # 构建当前窗口的KNN图
            knn_graph = build_knn_graph_batch(node_emb, k, device)
            knn_graphs.append(knn_graph)

    return window_knn_graphs

def get_windowed_g_list(t_list, graph_dict, seq_len):   #[pred_t,pred_t]
    # 获取所有时间点并排序
    times = sorted(graph_dict.keys())
    windowed_g_list = []
    for target_t in t_list:
        # 找到目标时间的索引, 前seq_len天
        target_idx = times.index(target_t)
        
        # 计算窗口起点
        start_idx = max(0, target_idx - seq_len)
        
        # 提取连续时间窗口（不重复且保持顺序）
        window_times = times[start_idx : target_idx]  # [start, target-1]
        
        # 直接获取对应图（不进行去重）
        window_graphs = [graph_dict[t] for t in window_times]
        
        windowed_g_list.append(window_graphs)
    
    return windowed_g_list

def generate_adv_graph_list(windowed_g_list, model, t_list, delta_A=3, delta_E=5, lr=0.001, epochs=10, attack_ratio=0.4):
    
    model.eval()
    torch.backends.cudnn.enabled = False                # 局部禁用 cuDNN
    device = next(model.parameters()).device
    # ---------- 冻结模型参数 ----------
    for p in model.parameters():
        p.requires_grad_(False)

    # 预处理：深拷贝并转移到设备
    originals = [[g.clone().to(device) for g in window] for window in windowed_g_list]
    adv_windows = [[g.clone().to(device) for g in window] for window in windowed_g_list]
    
    # 预计算原始编码（仅需一次）
    with torch.no_grad():
        Z_orig = model(t_list, originals)
    
    # 批量处理所有窗口中的图
    for win_idx, window in enumerate(originals):
        num_graphs = len(window)
        attack_num = int(num_graphs * attack_ratio)
        selected_indices = random.sample(range(num_graphs), attack_num)
        for g_idx in selected_indices:
            orig_g = window[g_idx]
            adv_g = adv_windows[win_idx][g_idx]
            A = adv_g.adjacency_matrix().to_dense().clone().detach().requires_grad_(True)
            E = adv_g.edata['text_emb'].clone().detach().requires_grad_(True)
            #构小图
            orig_g_cpu = orig_g.cpu()
            A_cpu = adv_g.adjacency_matrix().to_dense().clone().detach().cpu()
            global_ids = orig_g_cpu.ndata['id']         # 形状 [num_nodes]
            global_norms = orig_g_cpu.ndata['norm']    # 从大图获取norm
            
            optimizer = torch.optim.Adam([A, E], lr=lr)
            
            # ---------- 批量梯度计算 ----------
            for _ in range(epochs):
                optimizer.zero_grad()
                
                # 动态构建对抗图（避免复制整个窗口）
                mask_edge = (A_cpu > 0.5) & (A_cpu <= 1.0)
                src, dst = torch.where(mask_edge)
                tmp_g = dgl.graph((src, dst), num_nodes=orig_g_cpu.num_nodes()).to(device)
                
                # 将局部节点索引转换为全局ID
                edge_ids = orig_g_cpu.edge_ids(src, dst)
                
                # 继承边属性（eid和type）
                tmp_g.edata['eid'] = orig_g_cpu.edata['eid'][edge_ids].to(device)
                tmp_g.edata['type'] = orig_g_cpu.edata['type'][edge_ids].to(device)
                tmp_g.edata['rel_type'] = tmp_g.edata['type'].type(torch.int32)

                tmp_g.ndata['id'] = global_ids.to(device)
                tmp_g.ndata['norm'] = global_norms.to(device)
                tmp_g.edata['text_emb'] = E[:tmp_g.num_edges()]
                
                # 原位替换目标图（减少内存拷贝）
                temp_windows = [w[:] for w in originals]  # 浅拷贝窗口结构
                temp_windows[win_idx][g_idx] = tmp_g
                
                # 前向传播（只计算必要梯度）
                Z_adv = model(t_list, temp_windows)
                loss = -F.mse_loss(Z_adv, Z_orig)
                loss.backward()
                
                # ---------- 高效结构扰动 ----------
                with torch.no_grad():
                    grad_A = A.grad
                    if grad_A is not None:
                        # 边添加：向量化操作
                        add_mask = (grad_A > 0) & (A < 0.5)
                        add_scores = grad_A[add_mask]
                        if add_scores.numel() > 0:
                            topk = torch.topk(add_scores, min(delta_A, add_scores.size(0)))
                            A.data[add_mask] = torch.where(
                                torch.isin(torch.arange(add_mask.sum()), topk.indices), 1.0, A[add_mask]
                            )
                        
                        # 边删除：向量化操作
                        del_mask = (grad_A < 0) & (A > 0.5)
                        del_scores = -grad_A[del_mask]
                        if del_scores.numel() > 0:
                            topk = torch.topk(del_scores, min(delta_A, del_scores.size(0)))
                            A.data[del_mask] = torch.where(
                                torch.isin(torch.arange(del_mask.sum()), topk.indices), 0.0, A[del_mask]
                            )
                
                # ---------- 高效特征扰动 ----------
                if E.grad is not None:
                    grad_E = E.grad
                    abs_grad = torch.abs(grad_E)
                    flat_grad = abs_grad.view(-1)
                    topk = torch.topk(flat_grad, min(delta_E, flat_grad.size(0)))
                    E.data.view(-1)[topk.indices] += lr * torch.sign(grad_E).view(-1)[topk.indices]
                
                optimizer.step()
            
            # ---------- 最终图构建 ----------
            src, dst = torch.where(A > 0.5)
            src_cpu, dst_cpu = src.cpu(), dst.cpu()
            final_g = dgl.graph((src, dst), num_nodes=orig_g.num_nodes()).to(device)

            edge_ids = orig_g_cpu.edge_ids(src_cpu, dst_cpu)  # 必须是 orig_g_cpu 中存在的边
            final_g.edata["eid"] = orig_g_cpu.edata["eid"][edge_ids].to(device)
            final_g.edata["type"] = orig_g_cpu.edata["type"][edge_ids].to(device)
            final_g.edata["rel_type"] = final_g.edata["type"].type(torch.int32)  # 自动转换类型
            final_g.ndata["id"] = orig_g_cpu.ndata["id"].to(device)
            final_g.ndata["norm"] = orig_g_cpu.ndata["norm"].to(device)

            final_g.edata['text_emb'] = E.detach()[:final_g.num_edges()]
            adv_windows[win_idx][g_idx] = final_g

    torch.backends.cudnn.enabled = True  # 恢复原始状态

    return adv_windows

def event_contrastive_loss(z1, z2, temperature=0.5):
    """
    边级别对比学习损失函数（支持3D输入）
    输入参数：
        z1: 视图1的边表示 [batch_size, num_edges, dim]
        z2: 视图2的边表示 [batch_size, num_edges, dim]
        temperature: 温度参数
    输出：
        loss: 对比损失标量值
    """
    batch_size, num_edges, feat_dim = z1.shape
    device = z1.device
    
    # 合并batch和edge维度
    z1 = z1.view(-1, feat_dim)  # [batch_size*num_edges, dim]
    z2 = z2.view(-1, feat_dim)
    
    # 归一化处理
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(z1, z2.T)/ temperature  # [B*N, B*N]
    
    # 构建正样本标签
    labels = torch.arange(batch_size * num_edges, dtype=torch.long, device=device)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

def get_rel_embedding(num_rels,node_in_feat,batch_g,feature):
        init_rel_embeds = torch.zeros(num_rels, node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(feature,batch_g_rel,dim=0,reduce="mean").cuda()  # group by mean
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]

        return init_rel_embeds

import torch

def compute_coverage_efficiency(logits, labels, eta):
    #print("eta:",eta)
    probs = torch.sigmoid(logits)          # (batch,num_rels)
    s_k = 1.0 - probs                      # (batch,num_rels)
    #print("s_k:",s_k)
    pred_sets = (s_k <= eta.item()).float() #pre_set
    #print("pred_sets:",pred_sets)

    nonzero_mask = labels != 0             # (batch, num_rel)
    batch_idx, rel_idx = torch.nonzero(nonzero_mask, as_tuple=True)
    num_sample = torch.nonzero(nonzero_mask, as_tuple=False).size(0)        #true sample

    covered = pred_sets[batch_idx.long(), rel_idx.long()]                   # (batch,num_rels)
    coverage_num = covered.float().sum().item()                             #cover true sample

    pred_set_sizes = pred_sets.sum(dim=1)                                   #pred set size 
    set_size = pred_set_sizes.mean().item()
    #print("set_size:",set_size)
    return coverage_num, set_size, num_sample

