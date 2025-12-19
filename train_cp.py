def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import numpy as np
import time
from utils import *
import os
from sklearn.utils import shuffle
from models import *
from data import *
from build_baseline import *
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader 
import json

parser = argparse.ArgumentParser(description='ACL_2026')
parser.add_argument("--dp", type=str, default="/data3/data/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--model", type=str, default='glean', help="model name")
parser.add_argument("--n-hidden", type=int, default=32, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--calib_lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--adversarial_lr", type=float, default=1e-3, help="adversarial learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='EG/cp', help="cp dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=20, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')
parser.add_argument("--n_layers", type=int, default=1, help='number of layers')
parser.add_argument("--text_emd_dim", type=int, default=768, help='text embedding dim')
parser.add_argument("--calib_model_train", type=str, default=True, help='need train')
#baseline
parser = add_baseline_argument(parser)

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()

print("cuda",use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

coverage_list = []
efficiency_list = []

iterations = 0 
while iterations < args.runs:  
    iterations += 1
    print('****************** iterations ',iterations,)
    
    if iterations == 1:
        print("loading data...")
        num_nodes, num_rels = utils.get_total_number(
            args.dp + args.dataset, 'stat.txt')
        full_ngh_finder = get_neighbor_finder(args, f'{args.dp}/{args.dataset}/', 'train.txt', 'valid.txt', 'calib_train.txt', 'calib_valid.txt', 'test.txt', args.uniform, num_nodes, num_rels)

        train_dataset_loader = DistData( 
            args.dp, args.dataset, num_nodes, num_rels, set_name='train')
        valid_dataset_loader = DistData(
            args.dp, args.dataset, num_nodes, num_rels, set_name='valid')

        calib_train_dataset_loader = DistData(
            args.dp, args.dataset, num_nodes, num_rels, set_name='calib_train')
        calib_valid_dataset_loader = DistData(
            args.dp, args.dataset, num_nodes, num_rels, set_name='calib_valid')

        test_dataset_loader = DistData(
            args.dp, args.dataset, num_nodes, num_rels, set_name='test')
 
        #MTG\TGN shuffle=False SeCoGD n_hidden 200 gpu=0
        #False
        train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_4)
        valid_loader = DataLoader(valid_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)

        calib_train_loader = DataLoader(calib_train_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)
        calib_valid_loader = DataLoader(calib_valid_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)

        test_loader = DataLoader(test_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)

        #build model
        n_calib = sum(1 for _ in open(args.dp + args.dataset + '/calib_train.txt', "r", encoding="utf-8"))
        gnn_model = build_model(args, num_nodes, num_rels, full_ngh_finder)
        calib_model = build_calib_model(args, num_nodes, num_rels, n_calib)

        gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.calib_model == 'ncpnet' or args.calib_model == 'cfgnn' or args.calib_model == 'ugnn':
            calib_optimizer = torch.optim.Adam(calib_model.parameters(), lr=args.calib_lr, weight_decay=args.weight_decay)

        gnn_model_state_file, gnn_model_graph_file, gnn_outf = model_info(args, gnn_model)
        calib_model_state_file, calib_model_graph_file, calib_outf = calib_model_info(args, calib_model, gnn_model)
        if use_cuda:
            gnn_model.cuda()
            calib_model.cuda()

    if os.path.exists(gnn_model_state_file):
        print(f"[INFO] Found existing gnn model checkpoint: {gnn_model_state_file}")
        checkpoint = torch.load(gnn_model_state_file)
        gnn_model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"[INFO] Loaded pretrained gnn model (epoch={start_epoch}). Skipping training.")
    else:
        #@torch.no_grad()
        bad_counter = 0
        loss_small =  float("inf")
        #gnn model train and valid
        try:
            print("start gnn model training...")
            for epoch in range(1, args.max_epochs+1):
                
                train_loss = model_train(args, gnn_model, full_ngh_finder, num_rels, gnn_optimizer, train_loader, train_dataset_loader, epoch, set_name = 'train')  
                valid_loss, p, recall, f1, f2, acc = model_evaluate(args, gnn_model, num_rels, valid_loader, valid_dataset_loader, set_name='valid')
                
                if valid_loss < loss_small:
                    loss_small = valid_loss
                    bad_counter = 0
                    print('save better model...')
                    torch.save({'state_dict': gnn_model.state_dict(), 'epoch': epoch, 'global_emb': None}, gnn_model_state_file)
                    # evaluate(test_loader, test_dataset_loader, set_name='Test')
                else:
                    bad_counter += 1
                if bad_counter == args.patience:
                    break
            print("training done")

        except KeyboardInterrupt:
            print('-' * 80)
            print('Exiting from training early, epoch', epoch)

    if args.calib_model == 'ncpnet' or args.calib_model == 'cfgnn':
        if os.path.exists(calib_model_state_file):
            print(f"[INFO] Found existing calin model checkpoint: {calib_model_state_file}")
            checkpoint = torch.load(calib_model_state_file)
            calib_model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"[INFO] Loaded pretrained calib model (epoch={start_epoch}). Skipping training.")
        else:
            #cp train and valid
            try:
                bad_counter = 0
                loss_small =  float("inf")
                print("start cp training...")
                for epoch in range(1, args.max_epochs+1):
                    train_loss = calib_train_model(args, gnn_model, calib_model, calib_optimizer, calib_train_loader, calib_train_dataset_loader, full_ngh_finder, epoch, set_name = 'calib_train')
                    valid_loss = calib_valid_model(args, gnn_model, calib_model, calib_valid_loader, calib_valid_dataset_loader, full_ngh_finder, set_name='calib_valid')
                    
                    if valid_loss < loss_small:
                        loss_small = valid_loss
                        bad_counter = 0
                        print('save better model...')
                        torch.save({'state_dict': calib_model.state_dict(), 'epoch': epoch, 'global_emb': None}, calib_model_state_file)
                        # evaluate(test_loader, test_dataset_loader, set_name='Test')
                    else:
                        bad_counter += 1
                    if bad_counter == args.patience:
                        break
                print("training done")

            except KeyboardInterrupt:
                print('-' * 80)
                print('Exiting from training early, epoch', epoch)
        # Load the best saved calib model.
        print("\nstart testing...")
        checkpoint = torch.load(calib_model_state_file, map_location=lambda storage, loc: storage)
        calib_model.load_state_dict(checkpoint['state_dict'])
        print("Using best calib_model epoch: {}".format(checkpoint['epoch']))
        
        # Load the best saved gnn model.
        checkpoint = torch.load(gnn_model_state_file, map_location=lambda storage, loc: storage)
        gnn_model.load_state_dict(checkpoint['state_dict'])
        print("Using best gnn_model epoch: {}".format(checkpoint['epoch']))

        #compute eta
        eta = calib_model_get_eta(args, gnn_model, calib_model, calib_valid_loader, calib_valid_dataset_loader, full_ngh_finder)          #valid set
    else:
        eta = calib_model_get_eta(args, gnn_model, calib_model, calib_valid_loader, calib_valid_dataset_loader, full_ngh_finder)
    
    coverage, efficiency = calib_test_model(args, gnn_model, calib_model, test_loader, test_dataset_loader, eta, set_name='calib_test')
    print(args)
    coverage_list.append(coverage)
    efficiency_list.append(efficiency)
 
print('finish training, results ....')
# save average results
coverage_list = np.array(coverage_list)
efficiency_list = np.array(efficiency_list)

coverage_avg, coverage_std = coverage_list.mean(0), coverage_list.std(0)
efficiency_avg, efficiency_std = efficiency_list.mean(0), efficiency_list.std(0)

print('--------------------')
print("Coverage: {:.4f}".format(coverage_avg))
print("Coverage_std: {:.4f}".format(coverage_std))
print("Efficiency: {:.4f}".format(efficiency_avg))
print("Efficiency_std: {:.4f}".format(efficiency_std))

# save results
result = 'Model: {}, Dataset: {}, Coverage: {:.4f}, Coverage_std: {:.4f}, Efficiency: {:.4f}, Efficiency_std: {:.4f}, lr: {:.5f}\n'.format(args.model, args.dataset, coverage_avg, coverage_std, efficiency_avg,efficiency_std, args.lr)
with open('/data3/src/results.csv','a') as fd:
    fd.write(result)