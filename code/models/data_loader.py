# -*- coding:utf-8 -*-
"""script to generate data for different models""" 

import numpy as np
import torch
import pandas as pd
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data.dataset import Dataset


class RobertaDataGenerator:
    def __init__(self, seq_pair_list, batch_size, tokenizer, embedding_dim, device):
        self.seq_pair_list = seq_pair_list
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.len = int(np.ceil(len(seq_pair_list) / float(self.batch_size)))
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        batch_data = self.seq_pair_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = []
        y = []
        sentIDs = []
        for item in batch_data:
            sentIDs.append(item[0])
            x.append(item[-1])
            # x.append(item[1])
            y.append(item[1])
        encoded_dict = self.tokenizer.batch_encode_plus(x,
                                                        add_special_tokens=True,
                                                        padding=True,
                                                        truncation=True,
                                                        max_length = self.embedding_dim,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_mask = encoded_dict['attention_mask'].to(self.device)
        # token_type_ids = encoded_dict['token_type_ids'].to(self.device)
        labels = torch.LongTensor(y).unsqueeze(1).to(self.device)

        # print(input_ids.size(), attention_mask.size(), labels.size())
        # return input_ids, token_type_ids, attention_mask, labels, sentIDs
        return input_ids, attention_mask, labels, sentIDs


class ConcateDataGenerator():
    def __init__(self, seq_pair_list, batch_size, tokenizer, embedding_dim, device):
        self.seq_pair_list = seq_pair_list
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.len = int(np.ceil(len(seq_pair_list) / float(self.batch_size)))
        self.embedding_dim = embedding_dim
        self.device = device
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        batch_data = self.seq_pair_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_sents = []
        x_lfs = []
        y = []
        sentIDs = []
        sents = []
        # y_real = []
        for item in batch_data:
            finalsent =  item[3] + ' [SEP] '
            # finalsent = '[CLS] ' + str(item[1]) + ' [SEP] '
            lfs = item[2]
            for ele in lfs:
                finalsent += str(ele)
                finalsent += ' '
            # finalsent += '[SEP]'
            sentIDs.append(item[0])
            x_sents.append(finalsent)
            # x_lfs.append(item[2])
            # y.append(_small2big(int(item['LABEL'])))
            y.append(int(item[1]))
            sents.append(item[3])
        encoded_dict = self.tokenizer.batch_encode_plus(x_sents,
                                                add_special_tokens=True,
                                                padding=True,
                                                truncation=True,
                                                max_length = self.embedding_dim,
                                                return_attention_mask=True,
                                                return_tensors='pt')
        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_mask = encoded_dict['attention_mask'].to(self.device)
        # token_type_ids = encoded_dict['token_type_ids'].to(self.device)
        # feature_inputs = torch.tensor(x_lfs, dtype=torch.float).to(self.device)
        labels = torch.LongTensor(y).unsqueeze(1).to(self.device)
        # real_labels = torch.LongTensor(y_real).to(self.device)
        
        # return input_ids, token_type_ids, attention_mask, labels, sentIDs
        # return input_ids, attention_mask, labels, sentIDs, sents
        # print(x_sents[0])
        # print(input_ids.size(), attention_mask.size(), labels.size())
        return input_ids, attention_mask, labels, sentIDs
    
