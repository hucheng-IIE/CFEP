import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from aggregators import *
from utils import *
from modules_f import *
import time
import math
import random
import itertools
import collections
from Generator import Generator
from Discriminator import Discriminator
import pickle

import logging
#t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

#mtg
from modules.cache import Cache
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.cache_updater import get_cache_updater
from modules.embedding_module import get_embedding_module

#tgn
# from tgn_modules.memory import Memory
# from tgn_modules.message_aggregator import get_message_aggregator
# from tgn_modules.message_function import get_message_function
# from tgn_modules.memory_updater import get_memory_updater
# from tgn_modules.embedding_module import get_embedding_module

from seco_modules.decoder import ConvTransE
from seco_modules.aggregator import Aggregator

# event forecasting
class MTG_model(torch.nn.Module):
  def __init__(self, neighbor_finder, dim, num_nodes, num_edges, device, max_pool, graph_dict, hist_wind=7, n_layers=2,
               n_heads=2, dropout=0.1, use_cache=True,
               message_dimension=100,
               cache_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="mean",
               cache_updater_type="gru",
               dyrep=False):
    super(MTG_model, self).__init__()

    self.dim = dim
    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.max_pool = max_pool
    self.hist_wind = hist_wind
    self.logger = logging.getLogger(__name__)

    node_features = np.random.rand(num_nodes, self.dim)
    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.n_node_features = self.node_raw_features.shape[1]
    self.edge_raw_features = torch.nn.Parameter(torch.zeros(num_edges, self.dim)).to(device)
    self.n_nodes = num_nodes
    self.n_edge_features = dim
    self.num_rels = num_edges
    self.embedding_dimension = dim
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type

    self.use_cache = use_cache
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.entity_cache = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    self.threshold = 0.5
    self.out_func = torch.sigmoid

    if self.use_cache:
      self.entity_cache_dimension = dim
      raw_message_dimension = 2 * self.entity_cache_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.entity_cache = Cache(n_nodes=self.n_nodes,
                           cache_dimension=self.entity_cache_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.entity_cache_updater = get_cache_updater(module_type=cache_updater_type,
                                               cache=self.entity_cache,
                                               message_dimension=message_dimension,
                                               cache_dimension=self.entity_cache_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 cache=self.entity_cache,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_cache=use_cache,
                                                 n_neighbors=self.n_neighbors)

    if self.use_cache:
      self.rel_raw_message_dimension = 2 * self.entity_cache_dimension + dim + \
                              self.time_encoder.dimension
      self.rel_cache = Cache(n_nodes=num_edges,
                           cache_dimension=self.entity_cache_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.rel_message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.rel_message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=self.rel_raw_message_dimension,
                                                   message_dimension=self.rel_raw_message_dimension)
      self.rel_cache_updater = get_cache_updater(module_type=cache_updater_type,
                                               cache=self.rel_cache,
                                               message_dimension=self.rel_raw_message_dimension,
                                               cache_dimension=self.entity_cache_dimension,
                                               device=device)

    self.sentence_size = 768
    self.text_embedding_size = dim
    self.textEmbeddingLayer = torch.nn.Linear(self.sentence_size, self.text_embedding_size)
    self.memory = Memory(n_nodes=self.n_nodes, n_rels=self.num_rels, memory_dimension=dim, input_dimension=dim, device=device)
    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features, 
                                     1)
    #dim
    self.gnn_model = CompGCN(h_dim=dim, num_ents=self.n_nodes, num_rels=num_edges, sentence_size=self.sentence_size,  dropout=dropout, seq_len=hist_wind, maxpool=1, use_edge_node=0, use_gru=1, attn='', device=self.device)
    self.gnn_model.graph_dict = graph_dict
    # self.init_weights()
     

  def init_weights(self):
      for p in self.parameters():
          if p.data.ndimension() >= 2:
              nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
          else:
              stdv = 1. / math.sqrt(p.size(0))
              p.data.uniform_(-stdv, stdv)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
    source_nodes = np.array(source_nodes)
    destination_nodes = np.array(destination_nodes)
    edge_times = np.array(edge_times)
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])
    rels = np.concatenate([edge_idxs])

    cache = None
    time_diffs = None
    if self.use_cache:
      cache = self.entity_cache.get_cache(list(range(self.n_nodes)))
      last_update = self.entity_cache.last_update
      rel_cache = self.rel_cache.get_cache(list(range(self.num_rels)))

      ### Compute differences between the time the cache of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module

    if len(edge_times) > 0:
        node_embedding = self.embedding_module.compute_embedding(cache=cache,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:]

        unique_nodes = np.unique(nodes)
        unique_node_embedding = self.embedding_module.compute_embedding(cache=cache,
                                                                 source_nodes=unique_nodes,
                                                                 timestamps=np.array([timestamps[0] for i in range(unique_nodes.shape[0])]),
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        unique_rels = np.unique(edge_idxs)
        unique_rel_cache = self.rel_cache.get_cache(unique_rels)
        self.memory.update_memory(unique_node_embedding, unique_rel_cache, unique_nodes, unique_rels)

        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
        unique_rels, rel_id_to_messages = self.get_rel_raw_messages(destination_nodes,
                                                                        destination_node_embedding,
                                                                        source_nodes,
                                                                        source_node_embedding,
                                                                        edge_times, edge_idxs)

        self.entity_cache.store_raw_messages(unique_sources, source_id_to_messages)
        self.entity_cache.store_raw_messages(unique_destinations, destination_id_to_messages)
        self.rel_cache.store_raw_messages(unique_rels, rel_id_to_messages)
  
        self.update_entity_cache(positives, self.entity_cache.messages)
        self.update_rel_cache(rels, self.rel_cache.messages)

        self.entity_cache.clear_messages(positives)
        self.rel_cache.clear_messages(rels)
        
    all_nodes = np.unique(list(range(self.n_nodes)))
    all_nodes_cache = self.entity_cache.get_cache(all_nodes)
    all_nodes_memory = self.memory.get_nodes_memory(all_nodes)
    all_nodes_embeddings = torch.cat([all_nodes_cache, all_nodes_memory], dim=1)

    all_rels = np.unique(list(range(self.num_rels)))
    all_rels_cache = self.entity_cache.get_cache(all_rels)
    all_rels_memory = self.memory.get_rels_memory(all_rels)
    all_rels_embeddings = torch.cat([all_rels_cache, all_rels_memory], dim=1)


    return all_nodes_embeddings, all_rels_embeddings

  def predict(self, source_nodes, destination_nodes,  edge_idxs, edge_times, time_idx, n_neighbors=20):
    all_nodes_cache, all_rels_cache = self.compute_temporal_embeddings(
    source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors)
    t_list = [time_idx]
    #t_list = torch.tensor(time_idx + self.hist_wind)
    #print(t_list)
    #pred = self.gnn_model(t_list, all_nodes_cache, all_rels_cache)
    pred = self.gnn_model(t_list, all_nodes_cache, all_rels_cache)
    
    return pred

  def evaluate(self, pred, true_prob_r):
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]
        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]

        return nonzero_true_rel_idx, nonzero_prob_rel_idx

  def update_entity_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the cache with the aggregated messages
    self.entity_cache_updater.update_cache(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def update_rel_cache(self, rels, rel_messages):
    # Aggregate messages for the same relation types
    unique_rels, unique_messages, unique_timestamps = \
      self.rel_message_aggregator.aggregate(
        rels,
        rel_messages)
    if len(unique_rels) > 0:
      unique_messages = self.rel_message_function.compute_message(unique_messages)

    # Update the cache with the aggregated messages
    self.rel_cache_updater.update_cache(unique_rels, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_cache, updated_last_update = self.entity_cache_updater.get_updated_cache(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_cache, updated_last_update

  def get_updated_rel_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_rels, unique_rel_messages, unique_timestamps = \
      self.rel_message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_rels) > 0:
      unique_rel_messages = self.rel_message_function.compute_message(unique_rel_messages)

    updated_rel_cache, updated_rel_last_update = self.rel_cache_updater.get_updated_cache(unique_rels,
                                                                                 unique_rel_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_rel_cache, updated_rel_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    # rel_cache = self.edge_raw_features[edge_idxs]
    rel_cache = self.rel_cache.get_cache(edge_idxs)

    source_cache = source_node_embedding
    destination_cache = destination_node_embedding

    source_time_delta = edge_times - self.entity_cache.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_cache, destination_cache, rel_cache,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages


  def get_rel_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    source_cache = source_node_embedding
    destination_cache = destination_node_embedding

    source_time_delta = edge_times - self.entity_cache.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    sentence_embeddings = get_sentence_embeddings(self.gnn_model.graph_dict, source_cache, edge_times, device=self.device)
    text_embeddings = self.textEmbeddingLayer(sentence_embeddings)

    rel_message = torch.cat([source_cache, destination_cache, text_embeddings,
                                source_time_delta_encoding], dim=1)
    messages = defaultdict(list)
    unique_rels = np.unique(source_nodes)

    for i in range(len(edge_idxs)):
      messages[edge_idxs[i]].append((rel_message[i], edge_times[i]))

    return unique_rels, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder   
class glean_model(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn=''):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.rand(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.rand(num_ents, h_dim))

        self.word_embeds = None
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        self.word_graph_dict = None
        self.graph_dict = None

        self.aggregator= aggregator_event_glean(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, attn)
        if use_gru:
            self.encoder = nn.GRU(3*h_dim, h_dim, batch_first=True)
        else:
            self.encoder = nn.RNN(3*h_dim, h_dim, batch_first=True)
        self.linear_r = nn.Linear(h_dim, num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r): 
        pred, idx, _ = self.__get_pred_embeds(t_list)
        # pred = self.out_func(pred)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def __get_pred_embeds(self, t_list):
        #sorted_t, idx = t_list.sort(0, descending=True)
        embed_seq_tensor, len_non_zero = self.aggregator(t_list, self.ent_embeds, 
                                    self.rel_embeds, self.word_embeds, self.graph_dict,
                                    self.word_graph_dict, 
                                    self.ent_map, self.rel_map)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)

        pred = self.linear_r(feature).squeeze(1)
        pred = pred.float()
        return pred, feature
        
    def predict(self, t_list): 
        pred,  feature = self.__get_pred_embeds(t_list)
        # pred = self.out_func(pred)
        return pred

    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]         #index->class

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]        #index->class
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class CompGCN_RNN_model(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_gru = use_gru
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.graph_dict = None
        self.aggregator= aggregator_event_CompGCN(h_dim, dropout, num_ents, num_rels, seq_len, maxpool)
        if use_gru:
            self.encoder = nn.RNN(2*h_dim, h_dim, batch_first=True)
        else:
            self.encoder = nn.Linear(2*h_dim, h_dim)
        self.linear_r = nn.Linear(h_dim, num_rels)  #multi-label

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = nn.BCELoss()
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r): 
        pred, idx, _ = self.__get_pred_embeds(t_list)
        pred = self.out_func(pred)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def __get_pred_embeds(self, t_list):
        #sorted_t, idx = t_list.sort(0, descending=True)  
        embed_seq_tensor, len_non_zero = self.aggregator(t_list, self.ent_embeds,self.rel_embeds,self.graph_dict)
        if self.use_gru:
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,len_non_zero,batch_first=True)
            _, feature = self.encoder(packed_input)
        else:
            feature = self.encoder(embed_seq_tensor)
        feature = feature.squeeze(0)

        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,len_non_zero,batch_first=True)
        # _, feature = self.encoder(packed_input)
        # feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        
        pred = self.linear_r(feature).squeeze(1)
        pred = pred.float()
        return pred, feature
        
    def predict(self, t_list): 
        pred,  feature = self.__get_pred_embeds(t_list)
        #pred = self.out_func(pred)
        return pred 

    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        pred = self.out_func(pred)          #edit
        
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class PECF(nn.Module):
    def __init__(self, input_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1,n_layers=2,t_out=1,
                 rnn_layers=1,k=3,method='kmeans',num_s_rel=None,disc_func='lin',alpha=0,beta=0,text_emd_dim=None):
        super().__init__()
        self.h_dim = input_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # initialize rel and ent embedding
        self.node_embeds = nn.Parameter(torch.Tensor(num_ents, input_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, input_dim))
        self.init_weights()

        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        self.graph_dict = None
        self.aggregator= aggregator_event_PECF(input_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,rnn_layers,
                                          self.node_embeds,self.rel_embeds,text_emd_dim)
        if use_gru == 1:
            self.encoder = nn.GRU(input_dim, input_dim, batch_first=True)
        elif use_gru == 2:
            self.encoder = nn.LSTM(input_dim, input_dim, batch_first=True)
        else:
            self.encoder = nn.RNN(input_dim, input_dim, batch_first=True)
        
        #binary class
        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.decoder = nn.Linear(input_dim+1,1)

        self.threshold = 0.5
        self.out_func = torch.nn.Sigmoid()
        self.criterion = nn.BCELoss()

        #counterfactual
        self.k = k
        self.method = method
        self.target_rel = None
        self.rel_dict = None
        self.num_s_rel = num_s_rel
        self.disc_func = disc_func
        self.alpha = alpha
        self.beta = beta

    def init_weights(self):
        for p in self.parameters():
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
        else:
            stdv = 1. / math.sqrt(p.size(0))
            p.data.uniform_(-stdv, stdv)
    
    def forward(self, t_list, true_prob_r):
        #remove zero
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]
        true_prob_r = true_prob_r[nonzero_idx]
        #get history relation data
        rel_data = get_rel_data_list(self.rel_dict,t_list,true_prob_r)

        loss, _ , _= self.__get_pred_loss(t_list,rel_data,true_prob_r)
        return loss
    
    def __get_pred_loss(self, t_list, rel_data, y_true):
        #sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict)
        batch_adj = self.__get_edge_adj(t_list)
        #get relation embeds
        #(batch_size,num_rels,dim)
        feature = batch_relation_feature.cuda()
        #get counterfactual enhancement
        embeds_F,embeds_CF,A_F,A_CF = self.__get_embeds(feature,y_true,rel_data,batch_adj)

        pred_F_logit,X_F = self.__get_logit(embeds_F)
        pred_F = self.out_func(pred_F_logit)
        #pred_F = torch.argmax(pred_F_logit,dim=1)

        pred_CF,X_CF = self.__get_logit(embeds_CF)
        pred_CF = self.out_func(pred_CF)
        #get loss
        loss_F = self.criterion(pred_F,A_F)
        loss_CF = self.criterion(pred_CF,A_CF)

        loss_disc = self.__get_disc_loss(embeds_F,embeds_CF)
        loss = loss_F + self.alpha * loss_CF + self.beta * loss_disc

        return loss, pred_F, X_F
    
    def predict(self, t_list, y_data): 
        rel_data = get_rel_data_list(self.rel_dict,t_list,y_data)

        loss, pred_y, X_F = self.__get_pred_loss(t_list,rel_data,y_data)
        return loss, pred_y, X_F

    def evaluate(self, t, y_true):
        loss, pred = self.predict(t, y_true)
        prob_rel = torch.where(pred > self.threshold, 1, 0)
        # target
        y_true = y_true.view(-1)
        return y_true, prob_rel ,loss

    def __get_embeds(self,rel_embeds,A_F,rel_data,batch_adj):
        batch_embeds_F = []
        batch_embeds_CF = []
        A_CF = []
        #batch_size,feature
        for i in range(len(rel_embeds)):
            #(rel_nums,embeds)
            embeds = rel_embeds[i]
            adj = batch_adj[i]
            #every batch
            T,T_cf,all_rel_CF = get_counterfactual(adj,embeds,self.method,self.k,self.target_rel)
            rel_CF = [all_rel_CF[rel_index] for rel_index in rel_data[i]]
            rel_CF = list(set(rel_CF))
            A_F_item = A_F[i].cpu().detach().item()
            A_CF_item = get_A_CF_item(self.rel_dict,rel_CF,A_F_item)
            A_CF.append(A_CF_item)
           
            embeds_final_F = torch.cat((embeds,torch.from_numpy(T).cuda()),axis=1)
            embeds_final_CF = torch.cat((embeds,torch.from_numpy(T_cf).cuda()),axis=1)
            batch_embeds_F.append(embeds_final_F)
            batch_embeds_CF.append(embeds_final_CF)
        
        batch_embeds_F = torch.stack(batch_embeds_F).cuda()
        batch_embeds_CF = torch.stack(batch_embeds_CF).cuda()
        
        A_CF = torch.Tensor(A_CF).cuda()

        return batch_embeds_F.to(torch.float32),batch_embeds_CF.to(torch.float32),A_F,A_CF
    
    def __get_logit(self,embeds):
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        embeds_pooling = self.maxpooling(embeds).squeeze(1)
        #(batch_size,dim)->(batch_size,2)
        pred_logit = self.decoder(embeds_pooling).squeeze(1)
        #(batch_size)
        return pred_logit,embeds_pooling
    
    def __get_disc_loss(self,embeds_F,embeds_CF):
        sample_F,sample_CF = sample_relation(self.num_s_rel,embeds_F,embeds_CF)
        loss = calc_disc(self.disc_func,sample_F,sample_CF)
        return loss
    
    def __get_edge_adj(self,t_list):
        graph_dict = self.graph_dict
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
            else:
                time_list.append(torch.LongTensor(times[:length]))
        #each batch get graph
        #(batch_size,rel_num,self.num_ents)
        all_adj_matrices = []
        for index in range(len(t_list)):
            #init adj each rel get adj
            adj_matrices = []
            for _ in range(self.num_rels):
                #adj_matrices.append(np.zeros(self.num_ents))
                adj_matrices.append(np.zeros(self.num_ents))
            g_list = [graph_dict[tim.item()] for tim in time_list[index]]
            g = dgl.batch(g_list)
            rel_type = g.edata['rel_type']
            batch_g_uniq_rel = torch.unique(rel_type, sorted=True)
            for edge_id in batch_g_uniq_rel:
                selected_edges = (g.edata['rel_type'] == edge_id.item()).nonzero().squeeze(1).tolist()
                src_nodes_array,dst_nodes_array = g.edges()
                src_nodes = src_nodes_array[selected_edges].tolist()
                dst_nodes = dst_nodes_array[selected_edges].tolist()
                nodes = list(set(src_nodes+dst_nodes))
                adj_matrices[edge_id][nodes] += 1
            #each edge feature
            all_adj_matrices.append(adj_matrices)
        return np.array(all_adj_matrices)   
class DynamicGCN(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, vocab_size, dropout=0, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
        
        self.word_embeds = None
        self.word_graph_dict = None
        self.output = num_rels      #multi-label
        #self.out_func = nn.Sigmoid()
        #self.out_func = nn.Softmax()
        self.threshold = 0.5
        
        self.aggregator= aggregator_event_DynamicGCN(h_dim, dropout, num_ents, num_rels, vocab_size, seq_len, maxpool,self.output)
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
        
    def predict(self, t_list): 
        pred = self.aggregator(t_list, self.word_embeds, self.word_graph_dict)
        pred = pred.unsqueeze(0)        #batch,num_rels
        #pred = self.out_func(pred)
        return pred
    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, dim, num_nodes, num_edges, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=True,
               memory_update_at_start=False, message_dimension=128,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=True,
               use_source_embedding_in_message=True,
               dyrep=True):
    super(TGN, self).__init__()

    self.dim = dim
    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    node_features = np.random.rand(num_nodes, self.dim)
    edge_features = np.random.rand(num_edges, self.dim)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer_tgn(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)
    self.decoder = nn.Linear(self.n_nodes,num_edges)    #num_rels
    self.pred_func = nn.Sigmoid()
    self.threshold = 0.5
  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      # if self.dyrep:
      #   source_node_embedding = memory[source_nodes]
      #   destination_node_embedding = memory[destination_nodes]
      #   negative_node_embedding = memory[negative_nodes]
    return memory

  def predict(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    node_feature = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    # score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
    #                             torch.cat([destination_node_embedding,
    #                                        negative_node_embedding])).squeeze(dim=0)
    # pos_score = score[:n_samples]
    # neg_score = score[n_samples:]

    node_feature = self.affinity_score(node_feature).squeeze(1)
    pred = self.decoder(node_feature)
    pred = pred.unsqueeze(0)            #batch_num
    #pred = self.pred_func(pred)
    return pred

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = np.array(edge_times)
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
  def evaluate(self, pred, true_prob_r):
    prob_rel = self.pred_func(pred.view(-1))
    sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
    if torch.cuda.is_available():
        sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
    else:
        sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
    nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
    nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]
    # target
    true_prob_r = true_prob_r.view(-1)  
    nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
    sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
    nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]

    return nonzero_true_rel_idx, nonzero_prob_rel_idx
class TGCN(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, vocab_size, dropout=0, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
        
        self.word_embeds = None
        self.word_graph_dict = None
        self.output = num_rels          #num_rels 
        self.out_func = nn.Sigmoid()
        self.threshold = 0.5
        
        self.aggregator= aggregator_event_TGCN(h_dim, dropout, num_ents, num_rels, vocab_size, seq_len, maxpool,self.output)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
        
    def predict(self, t_list): 
        pred = self.aggregator(t_list, self.word_embeds, self.word_graph_dict)
        #pred = self.out_func(pred)
        
        return pred    

    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class RGCN_RNN_model(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_gru = use_gru
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.graph_dict = None
        self.aggregator= aggregator_event_tRGCN(h_dim, dropout, num_ents, num_rels, seq_len, maxpool)
        if use_gru:
            self.encoder = nn.RNN(2*h_dim, h_dim, batch_first=True)
        else:
            self.encoder = nn.Linear(2*h_dim, h_dim)
        self.linear_r = nn.Linear(h_dim, num_rels)

        self.threshold = 0.5
        #self.out_func = nn.Softmax(dim=1)
        self.out_func = torch.sigmoid
        self.criterion = nn.BCELoss()
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r): 
        pred, idx, _ = self.__get_pred_embeds(t_list)
        pred = self.out_func(pred)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def __get_pred_embeds(self, t_list):
        #sorted_t, idx = t_list.sort(0, descending=True)  
        embed_seq_tensor, len_non_zero = self.aggregator(t_list, self.ent_embeds,self.rel_embeds,self.graph_dict)
        if self.use_gru:
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,len_non_zero,batch_first=True)
            _, feature = self.encoder(packed_input)
        else:
            feature = self.encoder(embed_seq_tensor)
        feature = feature.squeeze(0)
        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        
        pred = self.linear_r(feature).squeeze(1)
        pred = pred.float()
        return pred, feature
        
    def predict(self, t_list): 
        pred,  feature = self.__get_pred_embeds(t_list)
        pred = self.out_func(pred)
        return pred
    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class SeCo(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 hyper_adj_ent, hyper_adj_rel, n_layers_hypergraph_ent, n_layers_hypergraph_rel, k_contexts,
                 h_dim, sequence_len, num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0, use_cuda=False,
                 gpu=0,graph_dict = None):
        super(SeCo, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.hyper_adj_ent = hyper_adj_ent
        self.hyper_adj_rel = hyper_adj_rel
        self.n_layers_hypergraph_ent = n_layers_hypergraph_ent
        self.n_layers_hypergraph_rel = n_layers_hypergraph_rel
        self.k_contexts = k_contexts
        self.num_ents_dis = num_ents * k_contexts
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu
        self.graph_dict = graph_dict
        self.threshold = 0.5

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.k_contexts, self.num_rels * 2, self.h_dim),
                                          requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(self.k_contexts, self.num_ents, self.h_dim),
                                              requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        #self.loss_e = torch.nn.CrossEntropyLoss()
        

        self.aggregators = torch.nn.ModuleList()  # one rgcn for each context
        for contextid in range(self.k_contexts):
            self.aggregators.append(Aggregator(
                h_dim,
                num_ents,
                num_rels * 2,
                num_bases,
                num_hidden_layers,
                encoder_name,
                self_loop=self_loop,
                dropout=dropout,
                use_cuda=use_cuda))

        self.dropout = nn.Dropout(dropout)

        self.time_gate_weights = torch.nn.ParameterList()
        self.time_gate_biases = torch.nn.ParameterList()
        for contextid in range(self.k_contexts):
            self.time_gate_weights.append(nn.Parameter(torch.Tensor(h_dim, h_dim)))
            nn.init.xavier_uniform_(self.time_gate_weights[contextid], gain=nn.init.calculate_gain('relu'))
            self.time_gate_biases.append(nn.Parameter(torch.Tensor(h_dim)))
            nn.init.zeros_(self.time_gate_biases[contextid])

        # GRU cell for relation evolving
        self.relation_gru_cells = torch.nn.ModuleList()
        for contextid in range(self.k_contexts):
            self.relation_gru_cells.append(nn.GRUCell(self.h_dim * 2, self.h_dim))
        # The number of expected features in the input x; The number of features in the hidden state h

        # decoder
        if decoder_name == "convtranse":
            self.decoders = torch.nn.ModuleList()
            for contextid in range(self.k_contexts):
                self.decoders.append(ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout))
        elif decoder_name == "Linear":
            self.decoders = torch.nn.ModuleList()
            self.decoders.append(nn.MaxPool2d((num_ents,h_dim)))
            self.decoders.append(nn.Linear(self.k_contexts,num_rels))   #num_rels
        self.out_func = nn.Sigmoid()

    def get_embs(self, g_list, use_cuda=True):

        # dynamic_emb entity embedding matrix H is global, but is normalized before every forward
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]

        ent_emb_each, rel_emb_each = [], []

        for contextid in range(self.k_contexts):
            ent_emb_context = self.h[contextid, :, :]
            rel_emb_context = self.emb_rel[contextid, :, :]
            for timid, g_each in enumerate(g_list):
                g = g_each[contextid].to(self.gpu)
                if len(g.r_len) == 0:
                    continue

                temp_e = ent_emb_context[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                    self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1], :]  # all entities related to a relation
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean

                x_input = torch.cat((rel_emb_context, x_input), dim=1)
                rel_emb_context = self.relation_gru_cells[contextid](x_input, rel_emb_context)  # new input hidden = h'
                rel_emb_context = F.normalize(rel_emb_context) if self.layer_norm else rel_emb_context

                curr_ent_emb_context = self.aggregators[contextid].forward(g, ent_emb_context,
                                                                           rel_emb_context)  # aggregated node embedding
                curr_ent_emb_context = F.normalize(curr_ent_emb_context) if self.layer_norm else curr_ent_emb_context

                time_weight = torch.sigmoid(
                    torch.mm(ent_emb_context, self.time_gate_weights[contextid]) + self.time_gate_biases[contextid])
                ent_emb_context = time_weight * curr_ent_emb_context + (1 - time_weight) * ent_emb_context

            ent_emb_each.append(ent_emb_context)
            rel_emb_each.append(rel_emb_context)
        if use_cuda:
          ent_emb_each = torch.stack(ent_emb_each).cuda()  # k, num_ent, h_dim
          rel_emb_each = torch.stack(rel_emb_each).cuda()  # k, num_rel * 2, h_dim

        if self.hyper_adj_ent is not None:
            ent_emb_each = self.forward_hyergraph_ent(ent_emb_each)
        if self.hyper_adj_rel is not None:
            rel_emb_each = self.forward_hyergraph_rel(rel_emb_each)

        return ent_emb_each, rel_emb_each

    def forward_hyergraph_ent(self, node_repr):
        node_repr = node_repr.transpose(0, 1).contiguous().view(-1, self.h_dim)
        for n in range(self.n_layers_hypergraph_ent):
            node_repr = F.normalize(self.dropout(self.hyper_adj_ent.cuda() @ node_repr)) + node_repr
        node_repr = node_repr.view(self.num_ents, self.k_contexts, self.h_dim).transpose(0, 1)
        return node_repr

    def forward_hyergraph_rel(self, rel_repr):
        rel_repr = rel_repr.transpose(0, 1).contiguous().view(-1, self.h_dim)
        for n in range(self.n_layers_hypergraph_rel):
            rel_repr = F.normalize(self.dropout(self.hyper_adj_rel.cuda() @ rel_repr)) + rel_repr
        rel_repr = rel_repr.view(self.num_rels * 2, self.k_contexts, self.h_dim).transpose(0, 1)
        return rel_repr

    def predict(self, t_list):
            
        # inverse_test_triplets = test_triplets[:, [2, 1, 0]]
        # inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels
        # all_triples = torch.cat((test_triplets, inverse_test_triplets))
        # all_contexts = torch.cat([test_contexts, test_contexts])  # [n_triplets, k_contexts]
        times = list(self.graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []
        # nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        # t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = times.index(tim)
            if self.sequence_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.sequence_len:length]))
                len_non_zero.append(self.sequence_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        # entity graph
        g_list = [self.graph_dict[tim.item()] for tim in unique_t]


        e_emb, r_emb = self.get_embs(g_list)
        pre_emb = F.normalize(e_emb, dim=-1) if self.layer_norm else e_emb  # [k, n_ents, h_dim]
        for i in range(len(self.decoders)):
            pre_emb = self.decoders[i](pre_emb)
            if i < len(self.decoders)-1:
                pre_emb = pre_emb.squeeze()
        pred = self.out_func(pre_emb)
        return pred

    def forward(self, glist, test_triples, use_cuda, test_contexts):
        """
        :param glist: [(g),..., (g)] len=valid_history_length
        :param test_triples: triplets to be predicted, ((s, r, o), ..., (s, r, o)) len=num_triplets
        :param use_cuda: use cuda or cpu
        :param test_contexts: [(1,0,0,0,0) onehot of contextid, ...] len=num_triplets
        :return: loss_ent
        """
        all_triples, final_score_ob = self.predict(glist, test_triples, use_cuda, test_contexts)
        loss_ent = self.loss_e(final_score_ob, all_triples[:, 2])

        return loss_ent
    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class GCN_model(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_gru = use_gru
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.graph_dict = None
        self.aggregator= aggregator_event_GCN(h_dim, dropout, num_ents, num_rels, seq_len, maxpool)
        if use_gru:
            self.encoder = nn.RNN(2*h_dim, h_dim, batch_first=True)
        else:
            self.encoder = nn.Linear(2*h_dim, h_dim)
        self.linear_r = nn.Linear(h_dim, num_rels)  #multi-label

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = nn.BCELoss()
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r): 
        pred, idx, _ = self.__get_pred_embeds(t_list)
        pred = self.out_func(pred)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def __get_pred_embeds(self, t_list):
        #sorted_t, idx = t_list.sort(0, descending=True)  
        embed_seq_tensor, len_non_zero = self.aggregator(t_list, self.ent_embeds,self.rel_embeds,self.graph_dict)
        if self.use_gru:
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,len_non_zero,batch_first=True)
            _, feature = self.encoder(packed_input)
        else:
            feature = self.encoder(embed_seq_tensor)
        feature = feature.squeeze(0)

        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,len_non_zero,batch_first=True)
        # _, feature = self.encoder(packed_input)
        # feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        
        pred = self.linear_r(feature).squeeze(1)
        pred = pred.float()
        return pred, feature
        
    def predict(self, t_list): 
        pred,  feature = self.__get_pred_embeds(t_list)
        pred = self.out_func(pred)
        return pred 

    def evaluate(self, t, true_prob_r):
        pred = self.predict(t)
        loss = soft_cross_entropy(pred, true_prob_r)
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class APEP_old(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        #self.word_embeds = None
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_AEPE(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.generator = Generator(h_dim, adversarial_lr)
        self.discriminator = Discriminator(h_dim, adversarial_lr)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)

        #adversarial loss
        fake_feature = self.generator(feature)
        discriminator_loss = self.discriminator.update(feature, fake_feature)
        generator_loss = self.generator.update(self.discriminator, fake_feature)

        loss = self.criterion(pred, true_prob_r[idx])

        #sum adversarial loss
        loss = loss + discriminator_loss + generator_loss
        
        return loss

    def __get_pred_embeds(self, t_list):
        sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class APEP(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
 
        #self.word_embeds = None
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_AEPE(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.generator = Generator(h_dim, adversarial_lr)
        self.discriminator = Discriminator(h_dim, adversarial_lr)
        self.temperature = 0.2

        self.gcl_mlp = nn.Sequential(
                nn.Linear(h_dim, h_dim),  
                nn.ReLU(),          
                nn.Linear(h_dim, h_dim)   
            )
        

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    

    def forward(self, t_list, windowed_g_list=None, Sem_graph=False):     #t_list [pre_t,pre_t]
        if windowed_g_list:
            _, event_feature = self.get_pred_embeds(windowed_g_list, Sem_graph)
        else:
            windowed_g_list = get_windowed_g_list(t_list, self.graph_dict, self.seq_len)
            _, event_feature = self.get_pred_embeds(windowed_g_list)

        return event_feature
 
    def get_pred_embeds(self, windowed_g_list, Sem_graph=False):
        batch_relation_feature = self.aggregator(windowed_g_list, Sem_graph)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)
        
        pred = self.linear_r(feature_pooling)

        return pred, feature

    def predict(self, t_list, true_prob_r): 
        sorted_t, idx = t_list.sort(0, descending=True) 
        windowed_g_list = get_windowed_g_list(t_list, self.graph_dict, self.seq_len)
        pred, feature = self.get_pred_embeds(windowed_g_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class GACN(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
 
        #self.word_embeds = None 
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_GACN(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.generator = Generator(h_dim, adversarial_lr)
        self.discriminator = Discriminator(h_dim, adversarial_lr)
        self.gcl_loss = gcl_loss

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r):
        pred_clean, idx_clean, feature_clean = self.__get_pred_embeds(t_list)                            #clean graph
        #pred_noise, idx_noise, feature_noise = self.__get_pred_embeds(t_liste)       #adversarial graph
        
        #adversarial loss
        fake_feature = self.generator(feature_clean)                                                     #adversiarial embedding
        discriminator_loss = self.discriminator.update(feature_clean, fake_feature)
        generator_loss = self.generator.update(self.discriminator, fake_feature)
        gcl_loss = self.gcl_loss(feature_clean,fake_feature)
        loss = self.criterion(pred_clean, true_prob_r[idx_clean])

        #sum adversarial loss
        loss = loss + discriminator_loss + generator_loss + gcl_loss
        
        return loss
 
    def __get_pred_embeds(self, t_list, add_graph_noise=False):
        sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict, add_graph_noise)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class GraphCL(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
 
        #self.word_embeds = None 
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_GraphCL(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.gcl_loss = gcl_loss

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r):
        pred_clean, idx_clean, feature_clean = self.__get_pred_embeds(t_list)                            #clean graph
        pred_noise, idx_noise, feature_noise = self.__get_pred_embeds(t_list,add_graph_noise=True)                           #adversarial graph
        
        #adversarial loss
        # fake_feature = self.generator(feature_clean)                                                     #adversiarial embedding
        # discriminator_loss = self.discriminator.update(feature_clean, fake_feature)
        # generator_loss = self.generator.update(self.discriminator, fake_feature)
        gcl_loss = self.gcl_loss(feature_clean, feature_noise)
        loss = self.criterion(pred_clean, true_prob_r[idx_clean])

        #sum adversarial loss
        loss = loss + gcl_loss
        
        return loss
 
    def __get_pred_embeds(self, t_list, add_graph_noise=False):
        sorted_t, idx = t_list.sort(0, descending=True)
        batch_relation_feature = self.aggregator(t_list, self.graph_dict, add_graph_noise)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
class GCA(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
 
        #self.word_embeds = None 
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_GraphCL(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.gcl_loss = gcl_loss

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r):
        pred_clean, idx_clean, feature_clean = self.__get_pred_embeds(t_list)                            #clean graph
        pred_noise, idx_noise, feature_noise = self.__get_pred_embeds(t_list,add_graph_noise=True)                           #adversarial graph
        
        #adversarial loss
        # fake_feature = self.generator(feature_clean)                                                     #adversiarial embedding
        # discriminator_loss = self.discriminator.update(feature_clean, fake_feature)
        # generator_loss = self.generator.update(self.discriminator, fake_feature)
        gcl_loss = self.gcl_loss(feature_clean, feature_noise)
        loss = self.criterion(pred_clean, true_prob_r[idx_clean])

        #sum adversarial loss
        loss = loss + gcl_loss
        
        return loss
 
    def __get_pred_embeds(self, t_list, add_graph_noise=False):
        sorted_t, idx = t_list.sort(0, descending=True)
        batch_relation_feature = self.aggregator(t_list, self.graph_dict, add_graph_noise)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
