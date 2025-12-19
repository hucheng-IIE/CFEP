import numpy as np
import random
import pandas as pd
import os
from collections import Counter

def generate_data_x(inPath, fileName, fileName2=None, fileName3=None):
  last_time = -1
  x_day = []
  x_data = []

  with open(os.path.join(inPath, fileName), 'r') as fr:
      #print(os.path.join(inPath, fileName))
      for line in fr:
          line_split = line.split()
          head = int(line_split[0])
          tail = int(line_split[2])
          rel = int(line_split[1])
          time = int(line_split[3])

          if time > last_time:
            for blank in range(last_time+1, time):
              x_data.append([])
            if last_time > -1:
              x_data.append(x_day)
            last_time = time
            x_day = []
          x_day.append((head, rel, tail, time))
  
  if fileName2 != None:
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])

            if time > last_time:
              for blank in range(last_time+1, time):
                x_data.append([])
              if last_time > -1:
                x_data.append(x_day)
              last_time = time
              x_day = []
            x_day.append((head, rel, tail, time))

  if fileName3 != None:
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])

            if time > last_time:
              for blank in range(last_time+1, time):
                x_data.append([])
              if last_time > -1:
                x_data.append(x_day)
              last_time = time
              x_day = []
            x_day.append((head, rel, tail, time))

  x_data.append(x_day)

  return x_data

def divide_data(x_data, y_data, lead_time, pred_wind):
  # shift = lead_time - 1
  # n = len(x_data)
  # x_data = x_data[:n-shift]
  # y_data = y_data[shift:]
  # assert len(x_data) == len(y_data)
  # print (len(x_data))
  cut_1 = 1795
  cut_2 = 2019
  if pred_wind > 1 or 1:
    y_data_window = [0 for i in range(len(y_data))]
    for i,y in enumerate(y_data):
      if y==1:
        for j in range(max(i-lead_time-pred_wind+2, 0), i-lead_time+2):
          y_data_window[j] = 1
    # print (y_data_window)
    y_data = y_data_window

  # x_train, x_valid, x_test = x_data[:cut_1], x_data[cut_1:cut_2], x_data[cut_2:]
  # y_train, y_valid, y_test = y_data[:cut_1], y_data[cut_1:cut_2], y_data[cut_2:]

  return x_data, y_data

def divide_data_online(x_data, y_data, lead_time, pred_wind):
  # shift = lead_time - 1
  # n = len(x_data)
  # x_data = x_data[:n-shift]
  # y_data = y_data[shift:]
  # assert len(x_data) == len(y_data)
  # print (len(x_data))
  sets = []
  sets.append([0, 412, 464, 516])
  # sets.append([0, 827, 930, 1033])
  # sets.append([0, 1240, 1395, 1550])
  # sets.append([0, 1653, 1860, 2067])
  # sets.append([0, 2068, 2326, 2584])

  if pred_wind > 1 or 1:
    y_data_window = [0 for i in range(len(y_data))]
    for i,y in enumerate(y_data):
      if y==1:
        for j in range(max(i-lead_time-pred_wind+2, 0), i-lead_time+2):
          y_data_window[j] = 1
    # print (y_data_window)
    y_data = y_data_window
  x_train_l, x_valid_l, x_test_l, y_train_l, y_valid_l, y_test_l = [],[],[],[],[],[]
  for s in sets:
      cut_1, cut_2, cut_3, cut_4 = s[0], s[1], s[2], s[3]
      x_train  = x_data[cut_1:cut_4]
      y_train = y_data[cut_1:cut_4]
      # x_train, x_valid, x_test = x_data[cut_1:cut_2], x_data[cut_2:cut_3], x_data[cut_3:cut_4]
      # y_train, y_valid, y_test = y_data[cut_1:cut_2], y_data[cut_2:cut_3], y_data[cut_3:cut_4]
      x_train_l.append(x_train)
      # x_valid_l.append(x_valid)
      # x_test_l.append(x_test)
      y_train_l.append(y_train)
      # y_valid_l.append(y_valid)
      # y_test_l.append(y_test)

  # return x_train_l, x_valid_l, x_test_l, y_train_l, y_valid_l, y_test_l
  return x_train_l, y_train_l


def generate_all(inPath, fileName, fileName2=None, fileName3=None):
  sources, destinations, timestamps = [], [], []
  with open(os.path.join(inPath, fileName), 'r') as fr:
    for line in fr:
        line_split = line.split()
        head = int(line_split[0])
        tail = int(line_split[2])
        rel = int(line_split[1])
        time = int(line_split[3])

        sources.append(head)
        destinations.append(tail)
        timestamps.append(time)
        
  with open(os.path.join(inPath, fileName2), 'r') as fr:
    for line in fr:
        line_split = line.split()
        head = int(line_split[0])
        tail = int(line_split[2])
        rel = int(line_split[1])
        time = int(line_split[3])

        sources.append(head)
        destinations.append(tail)
        timestamps.append(time)

  with open(os.path.join(inPath, fileName3), 'r') as fr:
    for line in fr:
        line_split = line.split()
        head = int(line_split[0])
        tail = int(line_split[2])
        rel = int(line_split[1])
        time = int(line_split[3])

        sources.append(head)
        destinations.append(tail)
        timestamps.append(time)

  return sources, destinations, timestamps

def get_batch_data(x_data):
  sources_batch = [x[0] for x in x_data]
  destinations_batch = [x[2] for x in x_data]
  edge_idxs_batch = [x[1] for x in x_data]
  timestamps_batch = [x[3] for x in x_data]
  #story_ids_batch = [x[4] for x in x_data]

  return sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch #story_ids_batch
