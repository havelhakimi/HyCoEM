import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from criterion import CLloss  # contrastive loss 




class PLM_hyp(nn.Module):
    def __init__(self, args, num_labels,idx2label):
        super().__init__()
        self.num_labels = num_labels
        self.args = args

        # Load pretrained transformer encoder (bert-base-uncased, roberta-base,google/electra-base-discriminator )
        self.encoder = AutoModel.from_pretrained(args.enc_type)
        self.config = self.encoder.config
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.num_labels * self.config.hidden_size, self.num_labels)



        # Attention mechanism to generate label-wise context vectors
        self.attention = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)

        # Contrastive learning loss module
        self.cl = CLloss(
            args,
            curv_init=args.curv_init,
            learn_curv=args.learn_curv,
            cl_temp=args.cl_temp,
            embed_dim=self.config.hidden_size
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        """m: batch_size, c: no of classes, h: hidden_size, s: token sequnece length"""
        # Pass inputs through the transformer encoder
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

        # Extract the last hidden layer output
        sequence_output = outputs.hidden_states[-1] # (m,s,h)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        text_enc=sequence_output[:, 0, :]

        #Compute attention weights and apply them to get a label-wise representation
        masks = torch.unsqueeze(attention_mask, 1)  # Shape: (m, 1, s)
        attention = self.attention(sequence_output).transpose(1, 2)  # Shape: (m, c, s)
        attention = attention.masked_fill(~masks.bool(), -np.inf)
        attention = F.softmax(attention, -1)  # Softmax over sequence length
        representation = attention @ sequence_output  # Shape: (m,c,h)
        # Flatten label-wise representations and classify
        rep_vec=representation.view(representation.shape[0], -1) # (m,c*h-->m*c)
        logits = self.classifier(rep_vec) 

        # Initialize loss components
        loss = None
        CE_loss = None

        if labels is not None:
            if self.args.cl_loss == 0:
                # Standard cross-entropy loss
                loss_fct = CrossEntropyLoss(reduction='mean')
                CE_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = CE_loss
            else:
                # Compute per-sample cross-entropy loss
                loss_fct = CrossEntropyLoss(reduction='none')
                CE_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # Compute contrastive loss on [CLS] token representation and attention weights as label features
                cl_loss = self.cl(sequence_output[:, 0, :], self.attention.weight, labels)

                # Weighted combination of CE and contrastive loss
                loss = self.args.cl_wt * torch.mean(cl_loss * CE_loss)

        return {
            'total_loss': loss,
            'CE_loss': CE_loss,
            'logits': logits,
            'rep_vec':rep_vec,
            'text_enc':text_enc
        }
