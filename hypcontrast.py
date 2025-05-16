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
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from util_functions import *
from model import PLM_hyp
from transformers import AutoConfig
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Class for training, validation, and testing
class HypContrast():
    def __init__(self, args, n_classes,idx2label):
        self.args = args

        # Load training, validation, and test datasets
        trainset = HyoEmoDataSet(args, 'train')
        validset = HyoEmoDataSet(args, 'valid')
        testset  = HyoEmoDataSet(args, 'test')

        # Create dataloaders
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate)
        self.valid_loader = DataLoader(validset, batch_size=256, shuffle=False, collate_fn=validset.collate)
        self.test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, collate_fn=testset.collate)

        # Set dataset and model config details
        args.n_samples, args.feat_dim = len(trainset), 768
        args.n_classes = n_classes
        #self.class_names = class_names 

        #self.model = PLM_hyp(args, num_labels=args.n_classes,)
        self.model = PLM_hyp(args, num_labels=args.n_classes,idx2label=idx2label)
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(f"Trainable: {trainable:,}, Non-trainable: {total - trainable:,}, Total: {total:,}")
        self.model.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        seed_torch(args.seed)
        
    def train_step(self, ith_epoch):
        self.model.train()
        train_pred, train_label = [], []
        total_loss = 0.0
        step = 0

        # Training progress bar
        p_bar = tqdm(self.train_loader, total=len(self.train_loader))
        for x, label in p_bar:
            step += 1
            input_ids = x['input_ids'].to(self.args.device)
            attention_mask = x['attention_mask'].to(self.args.device)
            labels = label.to(self.args.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output['total_loss']

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
            train_pred.extend(torch.argmax(output['logits'], dim=-1).tolist())
            train_label.extend(label.tolist())

            if step % 10 == 0:
                p_bar.set_description(f'train step {step} | loss={(total_loss/step):.4f}')

        train_acc = accuracy_score(train_pred, train_label)
        train_weighted_f1 = f1_score(train_pred, train_label, average='weighted')

        logging.info(f'''train | loss: {total_loss/step:.04f} acc: {train_acc:.04f}, f1: {train_weighted_f1:.04f}''')

        return {'loss': total_loss/step, 'train_acc': train_acc, 'train_weighted_f1': train_weighted_f1}

    def valid_step(self, ith_epoch):
        self.model.eval()
        valid_pred = None
        valid_label = []

        with torch.no_grad():
            for x, label in self.valid_loader:
                input_ids = x['input_ids'].to(self.args.device)
                attention_mask = x['attention_mask'].to(self.args.device)
                labels = label.to(self.args.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = output['logits']
                loss = output['total_loss']
                prediction = torch.argmax(logits, dim=-1)
                valid_pred = prediction if valid_pred is None else torch.cat([valid_pred, prediction])
                valid_label.extend(label.tolist())

        valid_pred = valid_pred.detach().cpu().numpy()
        valid_acc = accuracy_score(valid_pred, valid_label)
        valid_weighted_f1 = f1_score(valid_pred, valid_label, average='weighted')
        logging.info(f'''valid | loss: {loss:.04f} acc: {valid_acc:.04f}, f1: {valid_weighted_f1:.04f}''')

        return {
            'valid_loss': loss,
            'valid_pred': valid_pred,
            'valid_acc': valid_acc,
            'valid_weighted_f1': valid_weighted_f1
        }

    """def test_step(self, ith_epoch):
        self.model.eval()
        test_pred = None
        test_label = []

        with torch.no_grad():
            for x, label in self.test_loader:
                input_ids = x['input_ids'].to(self.args.device)
                attention_mask = x['attention_mask'].to(self.args.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                prediction = torch.argmax(logits, dim=-1)

                test_pred = prediction if test_pred is None else torch.cat([test_pred, prediction])
                test_label.extend(label.tolist())

        test_pred = test_pred.detach().cpu().numpy()
        test_acc = accuracy_score(test_pred, test_label)
        test_weighted_f1 = f1_score(test_pred, test_label, average='weighted')

        logging.info(f'''test | acc: {test_acc:.04f}, f1: {test_weighted_f1:.04f}''')

        return {
            'test_pred': test_pred,
            'test_acc': test_acc,
            'test_weighted_f1': test_weighted_f1
        }"""
    
    def test_step(self, ith_epoch):
        self.model.eval()
        test_pred = None
        test_label = []
        test_texts = []
        rep_vec=[]
        text_enc=[]

        with torch.no_grad():
            for x, label in self.test_loader:
                input_ids = x['input_ids'].to(self.args.device)
                attention_mask = x['attention_mask'].to(self.args.device)

                outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                prediction = torch.argmax(logits, dim=-1)

                test_pred = prediction if test_pred is None else torch.cat([test_pred, prediction])
                test_label.extend(label.tolist())
                test_texts.extend(input_ids)
                rep_vec.extend(outputs['rep_vec'])  
                text_enc.extend(outputs['text_enc'])


        test_pred = test_pred.detach().cpu().numpy()
        test_acc = accuracy_score(test_pred, test_label)
        test_weighted_f1 = f1_score(test_pred, test_label, average='weighted')

        logging.info(f'''test | acc: {test_acc:.04f}, f1: {test_weighted_f1:.04f}''')

        return {
            'test_pred': test_pred,
            'test_label': test_label,
            'test_texts': test_texts,
            'test_acc': test_acc,
            'test_weighted_f1': test_weighted_f1,
            'text_enc':text_enc,
            'rep_vec':rep_vec
        }

