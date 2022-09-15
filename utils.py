# coding: UTF-8
import os

import pandas as pd
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from twitter_preprocessor import TwitterPreprocessor

from text_cleaner import TextCleaner
CLS, SEP = '[CLS]','[SEP]'


def build_dataset(config):
    '''build train, dev, test set'''
    def load_dataset(path):
        contents = []
        label_dict = {"A": 0, "B": 1, "C": 2, "D": 3}
        content_key = config.content_key
        label_key = config.label_key
        data = pd.read_csv(path) # read data
        df = pd.DataFrame(data) # build DataFrame
        df[content_key] = df[content_key].apply(str)
        pad_size = config.pad_size # get padding size
        tokenizer = config.tokenizer # get the tokenizer
        text_cleaner = TextCleaner() # get text cleaner
        for idx in tqdm(range(len(df))):
            content = df[content_key].iloc[idx]
            label = df[label_key].iloc[idx]
            label = label_dict[label]
            p = TwitterPreprocessor(content)
            p.fully_preprocess() # clean the textual content of current instances
            clean_content = p.text
            if len(clean_content) == 0: # skip the instances with no words after preprocessing
                continue
            token = tokenizer.tokenize(clean_content)
            seq_len = len(token) # length of sequence before padding
            token_ids = tokenizer.encode(
                token,
                add_special_tokens = True,  # add [CLS] and [SEP] special tokens
                max_length = pad_size,
                padding = 'max_length',
                truncation=True,
            )
            if seq_len < pad_size:
                mask = [1] * seq_len + [0] * (pad_size - len(token)) #generate mask sequence
            else:
                mask = [1] * pad_size
            contents.append((token_ids, int(label), seq_len, mask))
        contents = contents[:200]
        return contents
    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return  train, dev, test


class DatasetIterator(object):
    '''Dataset Iterator to generate mini-batches for model training.
    Params:
        batches: input dataset
        batch_size: size of mini-batches
        device: computing device
    '''
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # record if number of batches is an int
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        '''convert data to tensor '''
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in data]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in data]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        '''get next batch'''
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    '''API for building dataset iterator'''
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """compute time difference"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
