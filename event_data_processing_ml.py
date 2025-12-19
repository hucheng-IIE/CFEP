import numpy as np
import random
import pandas as pd
import os
from utils import *

def get_distributions_ml(path, dataset, num_rels, seq_len):
    data, times = load_quadruples(path + dataset, 'train.txt', 'valid.txt', 'test.txt')
    time_l = list(set(data[:,-1]))
    time_l = sorted(time_l,reverse=False)
    y_data = []
    x_data = []
    for cur_t in time_l:
      x_day = [0 for _ in range(num_rels)]
      triples = get_data_with_t_ml(data, cur_t)
      true_r = np.zeros(num_rels)
      r_arr = triples[:,1]
      for r in r_arr:
          true_r[r] += 1
          x_day[r] += 1
      true_r = true_r / np.sum(true_r)
      true_r = [1 if x > 0 else 0 for x in true_r]
      y_data.append(true_r)
      x_data.append(x_day)
    
    #time->type count 2584 time
    x_cum_data = []
    for time in range(len(y_data)):
        x_cum_day = [0 for _ in range(num_rels)]
        if time < seq_len:
          for before_time in range(time):
            x_cum_day = [i+j for i,j in zip(x_cum_day,x_data[before_time])]
        else:
          for before_time in range(time-seq_len,time):
            x_cum_day = [i+j for i,j in zip(x_cum_day,x_data[before_time])]
        x_cum_data.append(x_cum_day)

    return x_cum_data, y_data 

def get_data_with_t_ml(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)