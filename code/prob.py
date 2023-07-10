# -*- coding: utf-8 -*-
from tqdm import tqdm
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import ast
from ferret import Benchmark

softmax = torch.nn.Softmax(dim=1)


class BertDataGenerator:
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

        # return input_ids, token_type_ids, attention_mask, labels, sentIDs
        return input_ids, attention_mask, labels, sentIDs
        

def json2list4eval(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            para_id = line["para_id"]
            sent_id = line["id"]
            sent_fi_id = str(para_id) + '_' + str(sent_id)    
            
            data.append([# int(line['sent_id']), \
                sent_fi_id, \
                # level, \
                # int(line['grade'] + line['label'] - 2), \
                # l2L_train(int(line['grade'] + line['label'] - 2)), \
                int(line['grade'] - 3), \
                # l2L(int(int(line['grade']) - 3)), \
                # l2L_train(int(line['grade'] - 2)), \
                # int(line['text_level']), \
                # int(line['label']), \
                line['sent'].strip()]) # label, input

    return data

def predict(infile, save_dir, outfile):
    test_data = json2list4eval(infile)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    # model.load_state_dict(torch.load(save_dir))
    model.eval()
    
    tokenizer_path = 'pretrained_models/bart-large/'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    bench = Benchmark(model, tokenizer)
    
    sum = len(test_data)
    acc, adj_acc = 0, 0

    # w = open(outfile, 'w', encoding='utf-8')
    jsondata = []
    
    for item_id, item in enumerate(tqdm(test_data)):
        sent_id = item[0]
        real = item[1]
        sent = item[-1]
        
        try:
            scores = bench.score(sent, return_dict=True)
            
            probs_str = ''
            for x in list(scores.values()):
                probs_str += str(x)
                probs_str += ' '

            jsondict = {'sent_id': sent_id, 'sent': sent, 'label': real, 'probs': probs_str}
            jsondata.append(jsondict)
        
        except:
            print('error')
            
    # w.close()
    jsondata_str = json.dumps(jsondata)
    jsonfile = open(outfile, 'w')
    jsonfile.write(jsondata_str)
    jsonfile.close()



def K_fold(indir, dataset, outdir):
    
    trained_models = {1: '1_9/', 2: '2_8/', 3: '3_4/', 4: '4_4/', 5: '5_3/', 6: '6_5/', 7: '7_7/', 8: '8_3/', 9: '9_9/', 10: '10_1/'}

    ids = [i for i in range(0, 10)]
    folds = 10
    for k_id in range(folds):
        
        trained_models_dir = indir + trained_models[k_id + 1]
        test_dataset = dataset + str(k_id) + '/' + 'test.json'
        result_file = outdir + str(k_id) + '.json'

        predict(test_dataset, trained_models_dir, result_file)




if __name__ == '__main__':
    K_fold(sys.argv[1], sys.argv[2], sys.argv[3])
