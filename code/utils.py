# -*- coding: utf-8 -*-

import os
import xlrd
import json
import numpy as np

import nltk
nltk.data.path.append('./nltk_data/')
from tqdm import tqdm



def _get_path(dir):
    # return file paths in the directory
    file_list = [root + '/' + f for root, dirs, files, in os.walk(dir) for f in files]

    return file_list


def json2list4train(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            sent_id = line['sent_id'] # cefr, cl2c
            # sent_id = line['id'] # wsj, clc
            sent = line['sent']
            label = int(line['label']) - 1 # cefr - 1, wsj not change
            # label = int(line['grade']) - 3 # clc - 3
            # label = int(line['level']) - 1 # cl2c - 1

            data.append([sent_id, \
                label, \
                sent]) # label, input
    return data


def json2list4eval(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            sent_id = line['sent_id'] # cefr, cl2c
            # sent_id = line['id'] # wsj, clc
            sent = line['sent']
            label = int(line['label']) - 1 # cefr - 1, wsj not change
            # label = int(line['grade']) - 3 # clc - 3
            # label = int(line['level']) - 1 # cl2c - 1

            data.append([sent_id, \
                label, \
                sent]) # label, input

    return data