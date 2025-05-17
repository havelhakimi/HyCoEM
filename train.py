import os
import json
import time
import pickle
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from pytorch_metric_learning import losses
from config import get_dicts
from util_functions import *
from hypcontrast import HypContrast
import argparse
import random


parser = argparse.ArgumentParser()

# Training configuration arguments
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--early-stop', type=int, default=10, help='Epoch before early stop.')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
parser.add_argument('--seed', type=int, default=3, help='seed for training')
parser.add_argument('--cl_loss', default=0, type=int, help='Contarstive loss (CL) in Lorentz Hyperbolic space')
parser.add_argument('--cl_temp', default=1, type=float, help='Temperature for CL loss')
parser.add_argument('--curv_init', type=int, default=1, help='Initial curvature (-k) for the Lorentz model, where k > 0.')
parser.add_argument('--learn_curv', action='store_true', default=True, help='Make scalars (text_alpha, label_alpha, and curv_init) learnable.')
parser.add_argument('--neg_sample',type=int,default=2,help='no of negative labels to sample for contrastive learning')
parser.add_argument('--pool', default=0, type=int, choices=[0, 1, 2], help='Optional single feedforward projection layer: 0 = no projection, 1 = projection with ReLU non-linearity, 2 = projection with GELU non-linearity')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--dataset', type=str, default='ED', choices=['go_emotion', 'ED','ED_easy_4', 'ED_hard_a', 'ED_hard_b', 'ED_hard_c', 'ED_hard_d'], help='A name for different runs.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 regularization) used in Adam optimizer.')
parser.add_argument('--enc_type', type=str, default='roberta-base', choices=['bert-base-uncased','roberta-base','google/electra-base-discriminator'], help='backbone.')
parser.add_argument('--device', type=str, default='cuda')




args = parser.parse_args()
print(args)
print(f'./exp/logs/{args.dataset}_b{args.batch_size}')
logging.basicConfig(filename=f'./exp/logs/{args.dataset}_b{args.batch_size}.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())



label2idx, idx2label = get_dicts(args.dataset)
num_classes = len(idx2label.items())
class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]


np.random.seed(args.seed)
torch.manual_seed(args.seed)
logging.info(f'Using device {args.device}, seed={args.seed}, training on {args.dataset} dataset.')



gm = HypContrast(args, num_classes,idx2label)
best_valid_weighted_f1, best_test_weighted_f1 = -1, -1
best_valid_loss = 1e5
train_loss, valid_loss = [], []
train_acc, valid_acc, test_acc = [], [], []
train_weighted_f1, valid_weighted_f1, test_weighted_f1 = [], [], []

total_train_time = 0
total_test_time = 0
early_stop_count = 0
for i in range(args.epochs):

    if early_stop_count >= args.early_stop:
        print("Early stop!")
        break
    #print()
    #print(f'Epoch {i} starts')
    start_time = time.time()
    train_log = gm.train_step(i)
    end_time = time.time()
    train_time = end_time - start_time
    total_train_time += train_time
    logging.info(f"Training time for this epoch: {train_time:.2f}")
    #print(f"Training time for this epoch: {train_time:.2f}")

    
    valid_log = gm.valid_step(i)
    test_start_time = time.time()
    test_log = gm.test_step(i)    
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    total_test_time += test_time
    logging.info(f"Inference time for this epoch: {test_time:.2f}")
    #print(f"Inference time for this epoch: {test_time:.2f}")

    train_acc.append(train_log['train_acc'])
    train_weighted_f1.append(train_log['train_weighted_f1'])
    train_loss.append(train_log['loss'])
    
    valid_acc.append(valid_log['valid_acc'])
    valid_weighted_f1.append(valid_log['valid_weighted_f1'])
    valid_loss.append(valid_log['valid_loss'])
    
    test_acc.append(test_log['test_acc'])
    test_weighted_f1.append(test_log['test_weighted_f1'])

    early_stop_count +=1
    if valid_log['valid_loss'] < best_valid_loss:
        early_stop_count = 0
        best_valid_loss = valid_log['valid_loss']
        test_acc_best_valid = test_log['test_acc']
        test_weighted_f1_best_valid = test_log['test_weighted_f1']
        logging.info(f"[valid loss new low] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")
        #print(f"[valid loss new low] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")        
    
    if valid_log['valid_weighted_f1'] > best_valid_weighted_f1:
        early_stop_count = 0
        best_valid_weighted_f1 = valid_log['valid_weighted_f1']
        best_valid_acc = valid_log['valid_acc']
        test_acc_best_valid = test_log['test_acc']
        test_weighted_f1_best_valid = test_log['test_weighted_f1']
        logging.info(f"[valid f1 new high] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")
        #print(f"[valid f1 new high] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")
        
    if test_log['test_weighted_f1'] > best_test_weighted_f1:
        best_test_weighted_f1 = test_log['test_weighted_f1']
        best_test_acc = test_log['test_acc']
        best_test_pred = test_log['test_pred']
        best_test_label = test_log['test_label']
        best_test_texts = test_log['test_texts']
        text_enc=test_log['text_enc']
        rep_vec=test_log['rep_vec']


    logging.info(f"[best] valid | acc: {best_valid_acc:.04f}, f1: {best_valid_weighted_f1:.04f}\n test | acc: {best_test_acc:.04f}, f1: {best_test_weighted_f1:.04f}")
    #print(f"[best] valid | acc: {best_valid_acc:.04f}, f1: {best_valid_weighted_f1:.04f}\n test | acc: {best_test_acc:.04f}, f1: {best_test_weighted_f1:.04f}") 
    #print(f'Epoch {i} ends')
#print()
logging.info(f"Average training time: {total_train_time/args.epochs}")
logging.info(f"Average inference time: {total_test_time/args.epochs}")
#print(f"Average training time: {total_train_time/args.epochs}")
#print(f"Average inference time: {total_test_time/args.epochs}")
print(f"test | acc: {best_test_acc:.04f}, f1: {best_test_weighted_f1:.04f}")


