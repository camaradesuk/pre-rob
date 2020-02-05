#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:53:39 2019

@author: qwang
"""


import os
import json
import random

from torchtext import data
import torchtext.vocab as vocab
import torch

#%%
class DataIterators(object):
    
    def __init__(self, args_dict):
        
        """
        Params:
            arg_dict: ...
        The dataset json is read and splitted into three jsons: "train.json", "val.json", "test.json".
        
        """
        
        self.args_dict = args_dict
            
        # Create data field
        self.ID = data.Field()
        self.LABEL = data.LabelField()
        
        if self.args_dict['net_type'] == 'han': 
            # nested sentence tokens
        	nest_field = data.Field(pad_token='<pad>', fix_length=self.args_dict['max_sent_len'])  # fix num of words in each sent (fix max_sent_len)
        	self.TEXT = data.NestedField(nest_field, fix_length=self.args_dict['max_doc_len'])  # fix num of sents (fix max_doc_len)
        else:
            self.TEXT = data.Field()   # word tokens 
            
 
    
    def split_and_save(self):
        """
        Read tokenized dataset in json format
        Shuffle and split data
        Write to separate json files
              
        """  
        data_json_path = self.args_dict['data_json_path']        
        #if os.path.exists(data_json_path) == False:
        #    raise('Data doesn't exist: {}'.format(os.path.basename(data_json_path)))
        
        dat = []
        try: 
            with open(data_json_path, 'r') as fin:
                for line in fin:
                    dat.append(json.loads(line))     
        except:
            print("Data doesn't exist: {}".format(os.path.basename(data_json_path)))
        
        
        # Cut sequence
        if self.args_dict['max_token_len'] != 0:
            for d in dat:
                if len(d['wordTokens']) > self.args_dict['max_token_len']:
                    d['wordTokens'] = d['wordTokens'][:self.args_dict['max_token_len']]
            
        random.seed(self.args_dict['seed'])
        random.shuffle(dat)
        
        train_size = int(len(dat) * self.args_dict['train_ratio'])
        val_size = int(len(dat) * self.args_dict['val_ratio'])
        
        train_list = dat[:train_size]
        val_list = dat[train_size : (train_size + val_size)]
        test_list = dat[(train_size + val_size):]
         
        
        data_dir = os.path.dirname(self.args_dict['data_json_path'])
        
        with open(os.path.join(data_dir, 'train.json'), 'w') as fout:
            for dic in train_list:     
                fout.write(json.dumps(dic) + '\n')
            
        with open(os.path.join(data_dir, 'val.json'), 'w') as fout:
            for dic in val_list:     
                fout.write(json.dumps(dic) + '\n')
        
        with open(os.path.join(data_dir, 'test.json'), 'w') as fout:
            for dic in test_list:     
                fout.write(json.dumps(dic) + '\n')   
          
        
        
    def create_data(self):
        """
        Create train/valid/test data
        
        """
        rob_item = self.args_dict['rob_name']
        
        
        if rob_item == 'random': rob_item = 'RandomizationTreatmentControl'
        if rob_item == 'blinded': rob_item = 'BlindedOutcomeAssessment'
        if rob_item == 'ssz': rob_item = 'SampleSizeCalculation'
        
        if self.args_dict['net_type'] == 'han':  	
            fields = {#'goldID': ('id', self.ID), 
        			  'label': ('label', self.LABEL), # rob_item: ('label', self.LABEL) for rob data
        			  'sentTokens': ('text', self.TEXT)}		
        else:
            fields = {#'goldID': ('id', self.ID), 
                       rob_item: ('label', self.LABEL), 
                       'wordTokens': ('text', self.TEXT)}
            

        train_data, valid_data, test_data = data.TabularDataset.splits(path = os.path.dirname(self.args_dict['data_json_path']),
                                                                       train = 'train.json',
                                                                       validation = 'val.json',
                                                                       test = 'test.json',
                                                                       format = 'json',
                                                                       fields = fields)
        return train_data, valid_data, test_data
        
    
    def load_embedding(self):
        
        embed_path = self.args_dict['embed_path']        
        custom_embedding = vocab.Vectors(name = os.path.basename(embed_path), 
                                         cache = os.path.dirname(embed_path))
        return custom_embedding
    
    
    def build_vocabulary(self, train_data, valid_data, test_data):
        self.ID.build_vocab(train_data, valid_data, test_data)
        self.LABEL.build_vocab(train_data)
        self.TEXT.build_vocab(train_data,
                              max_size = self.args_dict['max_vocab_size'],
                              min_freq = self.args_dict['min_occur_freq'],
                              vectors = self.load_embedding(),
                              unk_init = torch.Tensor.normal_)
    
    
    def create_iterators(self, train_data, valid_data, test_data):
        
        self.build_vocabulary(train_data, valid_data, test_data)
        
        ## CUDA
        if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
            device = torch.device('cuda') # torch.cuda.current_device()
        else:
            device = torch.device('cpu')
        
        
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = self.args_dict['batch_size'],
            device = device
        )
        
        return train_iterator, valid_iterator, test_iterator
        
##%% Instance   
#args_dict = {'seed': 1234,
#             'batch_size': 32,
#             'num_epochs': 2,
#             'train_ratio': 0.8,
#             'val_ratio': 0.1,
#             'max_vocab_size': 5000,
#             'min_occur_freq': 10,
#             'max_token_len': 5000,
#             'embed_dim': 200,
#             'dropout': 0.5,
#             
#             'exp_path': '/home/qwang/rob/src/cluster/exps',
#             'exp_name': 'han',
#             'rob_name': 'blinded',
#             'use_gpu': False,
#             'gpu_id': 'None',
#             
#             'args_json_path': None,
#             'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
#             'data_json_path': '/home/qwang/rob/amazon_tokens.json',
#             'use_cuda': False,
#             
#             'net_type': 'han',
#             'word_hidden_dim': 32,
#             'word_num_layers': 1,
#             
#             'sent_hidden_dim': 32,
#             'sent_num_layers': 1,
#             'max_sent_len': None,
#             'max_doc_len': None
#             }
#
#helper = DataIterators(args_dict = args_dict)
## Generate train/valid/test.json
#helper.split_and_save()
#train_data, valid_data, test_data = helper.create_data()   
#train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)
#
#print(helper.LABEL.vocab.stoi)  # {0: 0, 1: 1} ~= {'No': 0, 'Yes': 1}
#helper.TEXT.vocab.itos[:5]  # ['<unk>', '<pad>', 'the', 'i', 'and']
#
#len(helper.TEXT.vocab)  # 611
#len(helper.LABEL.vocab)  # 2
#
#helper.TEXT.pad_token  # '<pad>'
#helper.TEXT.unk_token  # '<unk>'
#helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1
#helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0
#helper.TEXT.vocab.vectors.shape  # [611, 20]
#
#class BatchWrapper:
#    def __init__(self, iterator, x_var, y_var):
#        self.iterator = iterator
#        self.x_var = x_var
#        self.y_var = y_var
#    
#    def __iter__(self):
#        for batch in self.iterator:
#            x = getattr(batch, self.x_var)
#            y = getattr(batch, self.y_var)
#            yield x, y
#            
#    
#train_batch = BatchWrapper(train_iterator, "text", "label")
#x_sent, y = next(train_batch.__iter__())
#x_sent.size()  # [20, 9 ,25] => [batch_size, max_doc_len, max_sent_len]
#
#d0_tokens = x_sent.permute(1,0)[0]
#s0_tokens = x_sent[0]   # 9 sents in doc 0. Each sent has 25 words
#print(len(x_sent[0][0]))  # sent 0 in doc 0
#
#
##%% NestedField
#import pprint
##from torchtext import data
#pp = pprint.PrettyPrinter(indent=4)
#
#minibatch = [
#     [['he', 'wants', 'a', 'banana'], ['I', 'am', 'sleepy'], ['hello']],
#     [['good'], ['hey', 'how', 'are', 'you']]
#]
## batch_size = 2
##   [doc 1]: doc_len = 3, sent_len = [4,3,1]
##   [doc 2]: doc_len = 2, sent_len = [1,4]
#
#nesting_field = data.Field(pad_token='<pad>', fix_length=None)  # fix num of words in each sent (fix sent_len)
#field = data.NestedField(nesting_field, fix_length=None)  # fix num of sents (fix doc_len)
#padded = field.pad(minibatch) 
#print(len(padded), len(padded[0]), len(padded[0][0])) # batch_size = 2, max_doc_len = 3, max_sent_len = 4
## >> Output
##    [[['he', 'wants', 'a', 'banana'],
##      ['I', 'am', 'sleepy', '<pad>'],
##      ['hello', '<pad>', '<pad>', '<pad>']],
##     [['good', '<pad>', '<pad>', '<pad>'],
##      ['hey', 'how', 'are', 'you'],
##      ['<pad>', '<pad>', '<pad>', '<pad>']]]
#
#
#nesting_field = data.Field(pad_token='<pad>', fix_length=3)  # fix num of words in each sent (fix sent_len)
#field = data.NestedField(nesting_field, fix_length=2)  # fix num of sents (fix doc_len)
#padded = field.pad(minibatch) 
#print(len(padded), len(padded[0]), len(padded[0][0])) # batch_size = 2, max_doc_len = 2, max_sent_len = 3
## >> Output
##    [[['he', 'wants', 'a'], 
##      ['I', 'am', 'sleepy']],
##     [['good', '<pad>', '<pad>'], 
##      ['hey', 'how', 'are']]]


