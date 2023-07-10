# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import logging

from models.AutoModel import K_fold

models = {'Roberta': K_fold}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Options: CNN, HAN, BERT, Augbert, Roberta")
    parser.add_argument("--state", type=str,
                        help="Options: classification, generation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=256,\
                        help="Length of embedding, Recommendations: 128, 256, 512")
    parser.add_argument("--dataset", type=str, default="../data/")
    parser.add_argument("--n_labels", type=int, default=12)
    parser.add_argument("--word2vec", type=str)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    # parser.add_argument("--best_epoch_file", type=str, default="")
    args = parser.parse_args()
    return args


def main(config):

    if config.state == 'classification':
        if config.model not in models:
            logging.error(f"The model you chosen is not supported yet.")
            return
        else:
            model = models[config.model](config.dataset, config.n_labels, \
                config.batch_size, config.epoch_num, config.lr, config.embedding_dim)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opt = get_args()
    main(opt)