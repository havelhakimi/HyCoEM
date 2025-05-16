
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import os
import lorentz
import math
from torch.nn import functional as F

# code for exponetial map projection and geodesic distance is in lorentz.py

pINF= 1000


class CLloss(torch.nn.Module):
    
    def __init__(self,args,curv_init=1,learn_curv=True,cl_temp=1,embed_dim=768,neg_samples=2):
        
        super().__init__()
        self.args=args
        self.cl_temp=cl_temp
        self.neg_samples=neg_samples
        # Initialize curvature for lorentz model
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        self._curv_minmax = {"max": math.log(curv_init * 10),"min": math.log(curv_init / 10),}

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.text_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.label_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

        self.dropout = nn.Dropout(0.1)
        self.pooler_fc = nn.Linear(768,768)


  
    def forward(self, text_embeddings, label_embeddings, target_labels):
        # m :batch size; h: fetaure size, c: no of labels
        """ text_embeddings: (m,h); label_embeddings: (c,h); target_labels: (m,)"""

        if self.args.pool==1:
            text_embeddings = F.relu(self.dropout(self.pooler_fc(text_embeddings)))
            label_embeddings = F.relu(self.dropout(self.pooler_fc(label_embeddings)))
        if self.args.pool==2:
            text_embeddings = F.gelu(self.dropout(self.pooler_fc(text_embeddings)))
            abel_embeddings = F.gelu(self.dropout(self.pooler_fc(label_embeddings)))



	    #  multiply by scalars
        text_embeddings = text_embeddings * self.text_alpha.exp()
        label_embeddings = label_embeddings* self.label_alpha.exp()

        # Project into hyperbolic space
        text_hyplr=  lorentz.exp_map0(text_embeddings,self.curv.exp())
        label_hyplr= lorentz.exp_map0(label_embeddings,self.curv.exp())

       
        batch_size = text_embeddings.size(0)


        # Calculate geodesic distance between text  and label embeddings
        dist_sim_matrix=lorentz.pairwise_dist(text_hyplr,label_hyplr,self.curv.exp())
        
        positive_labels = target_labels


        # Find hard negative labels
        hard_negative_labels = []

        for i, label in enumerate(positive_labels):
            hard_negative_labels_sample = []
            negative_similarities = dist_sim_matrix[i].clone()
            negative_similarities[label] = pINF  # Set positive labels' similarities to positive infinity

            sorted_indices = torch.argsort(negative_similarities, descending=False)
            # Find top-k hard negative labels for the label where k =self.args.neg_sample (A hyperparameter) 
            hard_negative_labels_sample.extend(sorted_indices[:self.args.neg_sample].tolist())
            hard_negative_labels.append(hard_negative_labels_sample)




        # Calculate contrastive loss

        loss = []
        
        for i in range(batch_size):
            zi = text_hyplr[i]
            pos_indices, neg_index = positive_labels[i], hard_negative_labels[i]

            # Calculate positive alignment score (text and its single positive label)
    

            pos_alignment_scores =-dist_sim_matrix[i, pos_indices]/self.cl_temp    
            pos_alignment_scores = pos_alignment_scores.unsqueeze(0)
            
            # Calculate negative alignment score (text and its top-k negative label)
            neg_alignment_scores = -dist_sim_matrix[i, neg_index]/self.cl_temp 

            denom= torch.cat([torch.exp(pos_alignment_scores), torch.exp(neg_alignment_scores)]).sum()
            pos_loss = -torch.log(torch.exp(pos_alignment_scores) /denom) 
            pos_loss=pos_loss.mean()
            loss.append(pos_loss)


        loss = torch.stack(loss).to(self.args.device)
        
        
        return loss






