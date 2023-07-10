import os
# os.environ["CURL_CA_BUNDLE"]=""
import sys
import random
from collections import defaultdict
from models.data_loader import RobertaDataGenerator
from models.data_loader import ConcateDataGenerator
# import os
sys.path.append('.')
import utils


import chardet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    confusion_matrix,
)

state = 'NIL'
model_name = 'bart_large/'

# pretrained_name = 'pretrained_models/bert-base-uncased/'
# pretrained_name = 'pretrained_models/bart-base/'
# pretrained_name = 'pretrained_models/roberta-base/'
# pretrained_name = 'pretrained_models/xlnet-base-cased/'
# pretrained_name = 'pretrained_models/electra-base-discriminator/'
pretrained_name = 'pretrained_models/bart-large/'
# pretrained_name = 'pretrained_models/roberta-large/'
# pretrained_name = 'pretrained_models/electra-large-discriminator/'



def _train(epoch_idx, num_epoch, model, train_data, optimizer, clip):
    model.train()
    epoch_loss = 0
    train_loop = tqdm(range(len(train_data)), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    train_loop.set_description('Epoch {}/{}'.format(epoch_idx + 1, num_epoch))
    
    adj_acc, acc = 0, 0
    sum = 0
    for i in train_loop:
        # input_ids, token_type_ids, attention_mask, labels, sentIDs = train_data[i]
        input_ids, attention_mask, labels, sentIDs = train_data[i]
        optimizer.zero_grad()
        outputs = model(input_ids,
                             # token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        epoch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loop.set_postfix(loss=epoch_loss / (i + 1))

        pred = torch.argmax(logits, dim=1).cpu().numpy()
        real = labels.squeeze(1).cpu().numpy()
        # print(pred.size(), real.size(), torch.sum(pred == real).item(), pred.size()[0])
        # correct += torch.sum(pred == real).item()
        # print(pred, real, type(pred), type(real))
        for i in range(len(pred)):
            sum += 1
            if pred[i] == real[i]:
                acc += 1
            if (real[i] - 1) <= int(pred[i]) <= (real[i] + 1):
                adj_acc += 1

    return epoch_loss / len(train_data), round(acc / sum, 4), round(adj_acc / sum, 4)


def _evaluate(model, val_data):
    model.eval()
    epoch_loss = 0
    result_dict = defaultdict(int)

    adj_acc, acc = 0, 0
    size = len(val_data) * 16
    sum = 0

    with torch.no_grad():
        for i in range(len(val_data)):
            # input_ids, token_type_ids, attention_mask, labels, sentIDs = val_data[i]
            input_ids, attention_mask, labels, sentIDs = val_data[i]
            outputs = model(input_ids,
                                 # token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            real = labels.squeeze(1).cpu().numpy()
            # print(pred.size(), real.size(), torch.sum(pred == real).item(), pred.size()[0])
            # correct += torch.sum(pred == real).item()
            # print(pred, real, type(pred), type(real))
            for i in range(len(pred)):
                sum += 1
                if pred[i] == real[i]:
                    acc += 1
                if (real[i] - 1) <= int(pred[i]) <= (real[i] + 1):
                    adj_acc += 1

    return epoch_loss / len(val_data), round(acc / sum, 4), round(adj_acc / sum, 4)


def _test(model, val_data, k_id, epoch_idx, test_file):
    model.eval()
    epoch_loss = 0
    result_dict = defaultdict(int)

    adj_acc, acc = 0, 0
    size = len(val_data) * 16
    sum = 0

    preds, reals = [], []
    w = open(test_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for i in range(len(val_data)):
            # input_ids, token_type_ids, attention_mask, labels, sentIDs = val_data[i]
            input_ids, attention_mask, labels, sentIDs = val_data[i]
            outputs = model(input_ids,
                                 # token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            real = labels.squeeze(1).cpu().numpy()
            # print(pred.size(), real.size(), torch.sum(pred == real).item(), pred.size()[0])
            # correct += torch.sum(pred == real).item()
            # print(pred, real, type(pred), type(real))
            for i in range(len(pred)):
                sum += 1
                w.write(str(sentIDs[i]) + '\t' + str(real[i]) + '\t' + str(pred[i]) + '\n')

                preds.append(pred[i])
                reals.append(real[i])

                if pred[i] == real[i]:
                    acc += 1
                if (real[i] - 1) <= int(pred[i]) <= (real[i] + 1):
                    adj_acc += 1
    w.close()

    macro_f1 = f1_score(reals, preds, average='macro')
    weighted_f1 = f1_score(reals, preds, average='weighted')
    precision = precision_score(reals, preds, average='weighted')
    recall = recall_score(reals, preds, average='weighted')
    qwk = cohen_kappa_score(reals, preds, weights='quadratic')
    adj_acc = utils.custom_adjacent_accuracy_score(reals, preds)
    mae = mean_absolute_error(reals, preds)

    # print(round(acc / sum, 4), round(adj_acc / sum, 4))

    return epoch_loss / len(val_data), macro_f1, round(acc / sum, 4), round(adj_acc / sum, 4), \
        weighted_f1, precision, recall, qwk, adj_acc, mae


def train(train_path, dev_path, test_path, pred_results_dir, n_labels, batch_size, epoch_num, lr, embedding_dim, k_id, dpr):
    # build a training model
    train_data_list = utils.json2list4train(train_path)
    dev_data_list = utils.json2list4eval(dev_path)
    test_data_list = utils.json2list4eval(test_path)
    
    random.shuffle(train_data_list)
    random.shuffle(dev_data_list)
    random.shuffle(test_data_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name,
                                                                # local_files_only=True,
                                                                num_labels=n_labels,
                                                                output_attentions=False,
                                                                output_hidden_states=False,
                                                                attention_probs_dropout_prob=dpr,
                                                                hidden_dropout_prob=dpr).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    train_data = RobertaDataGenerator(train_data_list, batch_size, tokenizer, embedding_dim, device) 
    dev_data = RobertaDataGenerator(dev_data_list, batch_size, tokenizer, embedding_dim, device)
    test_data = RobertaDataGenerator(test_data_list, batch_size, tokenizer, embedding_dim, device)

    # Concate
    # train_data = ConcateDataGenerator(train_data_list, batch_size, tokenizer, embedding_dim, device) 
    # dev_data = ConcateDataGenerator(dev_data_list, batch_size, tokenizer, embedding_dim, device)
    # test_data = ConcateDataGenerator(test_data_list, batch_size, tokenizer, embedding_dim, device)

    # lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # epoch_num = 10
    clip = 1

    best_train_loss, best_train_acc = 0, 0
    best_dev_loss, best_dev_acc = 9999, 0
    best_test_loss, best_test_acc = 9999, 0

    # best_train_adj_loss, best_train_adj_acc = 0, 0
    best_dev_adj_acc = 0
    best_test_adj_acc = 0
    best_epoch_num = 0

    best_macro_f1, best_weighted_f1, best_precision, best_recall, best_qwk, best_adj_acc, best_mae = 0, 0, 0, 0, 0, 0, 0

    for epoch_idx in range(epoch_num):
        # accuracy
        train_loss, train_acc, train_adj_acc = _train(epoch_idx, epoch_num, model, train_data, optimizer, clip)
        dev_loss, dev_acc, dev_adj_acc = _evaluate(model, dev_data)

        test_file = pred_results_dir + str(k_id) + '_' + str(epoch_idx) + '.dat'
        test_loss, test_macro_f1, test_acc, test_adj_acc, \
            weighted_f1, precision, recall, qwk, adj_acc, mae = _test(model, test_data, k_id, epoch_idx, test_file)

        print('train_loss, train_acc, test_loss, test_acc:', train_loss, train_acc, test_loss, test_acc)
        print('train_adj_acc, test_adj_acc:', train_adj_acc, test_adj_acc)
        
        if dev_loss < best_dev_loss:
            best_epoch_num = epoch_idx

            best_train_loss = train_loss
            best_train_acc = train_acc
            best_dev_loss = dev_loss
            best_dev_acc = dev_acc
            best_test_loss = test_loss
            best_test_acc = test_acc

            best_dev_adj_acc = dev_adj_acc
            best_test_adj_acc = test_adj_acc

            best_macro_f1 = test_macro_f1
            best_weighted_f1 = weighted_f1
            best_precision = precision
            best_recall = recall
            best_qwk = qwk
            best_adj_acc = adj_acc
            best_mae = mae

            # output_dir = './trained_models/clc/lert_large_' + str(batch_size) + '_' + str(lr) + '_' + str(epoch_num) + '/'
            # os.makedirs(output_dir, exist_ok=True)

            # para_dir = output_dir + str(k_id + 1) + '_' + str(epoch_idx) + '/'
            # os.makedirs(para_dir, exist_ok=True)

            # print("Saving model to %s" % output_dir)
            # model_to_save = model.module if hasattr(model, 'module') else model
            # model_to_save.save_pretrained(para_dir)
            # tokenizer.save_pretrained(para_dir)
        else:
            continue

        # print('train_loss, val_loss, train_acc, val_acc: ', train_loss, dev_loss, train_acc, dev_acc)
        # print('test_loss, test_acc', test_loss, test_acc)    
    
    # return best_epoch_num, best_train_loss, best_train_acc, best_dev_loss, best_dev_acc, best_test_loss, best_test_acc,\
    #     best_dev_adj_acc, best_test_adj_acc
    return best_epoch_num, best_train_loss, best_train_acc, best_dev_loss, best_dev_acc, best_test_loss, best_macro_f1, best_test_acc, \
        best_weighted_f1, best_precision, best_recall, best_qwk, best_adj_acc, best_mae


def K_fold(indir, n_labels, batch_size, epoch_num, lr, embedding_dim, dpr):
    # number fold
    fold_k = 1

    train_loss, train_acc = 0, 0
    dev_loss, dev_acc = 0, 0
    test_loss, test_acc = 0, 0

    train_adj_loss, train_adj_acc = 0, 0
    dev_adj_loss, dev_adj_acc = 0, 0
    test_adj_loss, test_adj_acc = 0, 0

    macro_f1, weighted_f1, precision, recall, qwk, adj_acc, mae = 0, 0, 0, 0, 0, 0, 0

    # write best_epoch of each fold for further automated bash
    # best_epoch_dir = 'Sta_best_epochs_eng/'
    best_epoch_dir = 'Best_epochs/WSJ/' + model_name
    os.makedirs(best_epoch_dir, exist_ok=True)
    outfile = best_epoch_dir + str(batch_size) + '_' + str(epoch_num) + '_' + str(lr) + \
                '_' + state + '_' + str(n_labels) + '_10fold.dat'
    w = open(outfile, 'w', encoding='utf-8')

    # pred_results_dir_last = 'Sta_pred_results_eng/'
    pred_results_dir_last = 'Pred_results/WSJ/' + model_name
    os.makedirs(pred_results_dir_last, exist_ok=True)
    pred_results_dir = pred_results_dir_last + str(batch_size) + '_' + str(epoch_num) + '_' + str(lr) + \
                '_' + state + '_' + str(n_labels) + '_10fold/'
    os.makedirs(pred_results_dir, exist_ok=True)

    
    ids = [i for i in range(0, 10)]
    for k_id in range(fold_k):

        train_paths, dev_path, test_path = [], '', ''
        
        if k_id != 9:
            for id in ids:
                if id == k_id:
                    dev_path = indir + '/' + str(id + 1) + '.json'
                if id == k_id + 1:
                    test_path = indir + '/' + str(id + 1) + '.json' 
                else:
                    train_paths.append(indir + '/' + str(id + 1) + '.json')
        
        if k_id == 9:
            dev_path = indir + '/' + '10' + '.json'
            test_path = indir + '/' + '1' + '.json'
            for id in range(2, 10):
                train_paths.append(indir + '/' + str(id) + '.json')

        best_epoch_num, train_loss_temp, train_acc_temp, dev_loss_temp, dev_acc_temp, test_loss_temp, test_macro_f1_temp, test_acc_temp, \
                test_f1_temp, test_precision_temp, test_recall_temp, test_qwk_temp, test_adj_acc_temp, test_mae_temp \
                 = train(train_paths, dev_path, test_path, pred_results_dir, n_labels, batch_size, epoch_num, lr, embedding_dim, k_id, dpr)
        print('train_acc, test_acc', train_acc_temp, test_acc_temp)
        
        train_loss += train_loss_temp
        train_acc += train_acc_temp
        dev_loss += dev_loss_temp
        dev_acc += dev_acc_temp
        test_loss += test_loss_temp
        test_acc += test_acc_temp

        macro_f1 += test_macro_f1_temp
        weighted_f1 += test_f1_temp
        precision += test_precision_temp
        recall += test_recall_temp
        qwk += test_qwk_temp
        adj_acc += test_adj_acc_temp
        mae += test_mae_temp

        # train_adj_loss += train_adj_loss_temp
        # train_adj_acc += train_adj_acc_temp
        # dev_adj_loss += dev_adj_loss_temp
        # dev_adj_acc += dev_adj_acc_temp
        # test_adj_loss += test_adj_loss_temp
        # test_adj_acc += test_adj_acc_temp

        w.write(str(k_id + 1) + '\t' + str(best_epoch_num) + '\n')

    print('{} fold Cross Validation RESULTS'.format(fold_k))
    print('train_loss_sum:%.4f' % (train_loss / fold_k), 'train_acc_sum:%.4f\n' % (train_acc / fold_k), \
          'dev_loss_sum:%.4f' % (dev_loss / fold_k), 'dev_acc_sum:%.4f' % (dev_acc / fold_k), \
          'test_loss_sum:%.4f' % (test_loss / fold_k), 'test_acc_sum:%.4f' % (test_acc / fold_k))

    print('f1: %.4f' % (weighted_f1 / fold_k), 'precision: %.4f' % (precision / fold_k), 'recall: %.4f' % (recall / fold_k), \
        'qwk: %.4f' % (qwk / fold_k), 'adj_acc: %.4f' % (adj_acc / fold_k), 'mae: %.4f' % (mae / fold_k))