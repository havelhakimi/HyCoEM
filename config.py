import os
import pickle
import argparse

all_dataset_list = ['go_emotion', 'ED','ED_easy_4', 'ED_hard_a', 'ED_hard_b', 'ED_hard_c', 'ED_hard_d']
ENCODER_TYPE = 'roberta-base' #bert-base-uncased ;roberta-base; google/electra-base-discriminator


def get_dicts(dataset):
    assert dataset in all_dataset_list
    if dataset == 'ED':
        label2idx = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
             'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
             'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13, 
             'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
             'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
             'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
             'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}
        
    elif dataset == 'go_emotion':
        label2idx = {'admiration': 0, 'amusement': 1, 'anger': 2,
              'annoyance': 3, 'approval': 4, 'caring': 5,
              'confusion': 6, 'curiosity': 7, 'desire': 8,
              'disappointment': 9, 'disapproval': 10, 'disgust': 11,
              'embarrassment': 12, 'excitement': 13, 'fear': 14,
              'gratitude': 15, 'grief': 16, 'joy': 17,
              'love': 18, 'nervousness': 19, 'optimism': 20,
              'pride': 21, 'realization': 22, 'relief': 23,
              'remorse': 24, 'sadness': 25, 'surprise': 26}
    

    elif dataset == 'ED_easy_4':
        label2idx = {'sad': 0, 'joyful': 1, 'angry': 2, 'afraid': 3}
    
    elif dataset == 'ED_hard_a':
        label2idx = {'anxious': 0, 'apprehensive': 1, 'afraid': 2, 'terrified': 3}
    
    elif dataset == 'ED_hard_b':
        label2idx = {'sad': 0, 'devastated': 1, 'sentimental': 2, 'nostalgic': 3}
    
    elif dataset == 'ED_hard_c':
        label2idx = {'angry': 0, 'ashamed': 1, 'furious': 2, 'guilty': 3}
    
    elif dataset == 'ED_hard_d':
        label2idx = {'anticipating': 0, 'excited': 1, 'hopeful': 2, 'guilty': 3}

    idx2label = {v: k for k, v in label2idx.items()}
    

        
    return label2idx, idx2label
    
