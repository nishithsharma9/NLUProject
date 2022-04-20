import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import pytorch_lightning as pl
import pickle
from collections import Counter

class NLU_dataset(Dataset):
    "Transform Bias Data into encoded tensor"
    def __init__(self, data, tokenizer, max_token_len: int = 512):
        """
        Args:
            :param data: pandas dataframe
            :param tokenizer: trained tokenizer from DeBERTa
            :param max_token_len: maximum length of tensor for each input tensor
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.toxic_id = tokenizer("[TOXIC]", return_tensors="pt").input_ids.flatten()[1]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        text = data_row.Text
        labels = data_row.Label

        encoding = self.tokenizer.encode_plus(text,
                                              add_special_tokens=True,
                                              max_length=self.max_token_len,
                                              return_token_type_ids=False,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt')

        return dict(text=text,
                    input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten() != self.toxic_id,
                    labels=torch.FloatTensor(labels))
                    
class NLU_DataModule(pl.LightningDataModule):
    '''
    Data Module for NLU classification task.
    train, val, test splits and transforms
    '''

    def __init__(self, train_df, test_df, tokenizer, batch_size: int = 12, max_token_len: int = 512):
        """
        Args:
            :param train_df: train dataset
            :param test_df: validation dataset
            :param tokenizer: trained tokenizer from 
            :param batch_size:  desired batch size
            :param max_token_len: maximum length of tensor for each input tensor
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.test_dataset = NLU_dataset(self.test_df, self.tokenizer, self.max_token_len)

    def setup(self, stage=None):
        """Encode the train & valid dataset (already splitted)"""
        self.train_dataset = NLU_dataset(self.train_df, self.tokenizer, self.max_token_len)
        self.test_dataset = NLU_dataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        """train set removes a subset to use for validation"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """validation set removes a subset to use for validation"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """test set"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class VOC_Data_Transform:
    def __init__(self, file: str = None):
        self.directory = os.path.abspath(os.getcwd())
        self.file = file

    def transform(self):
        # input files
        voc_testset = pd.read_excel(self.directory + '/voc_data/' + self.file + '.xlsx', dtype=str)
        voc_testset['VOC1'] = voc_testset.VOC1.str.replace('\n', ' ')
        voc_testset['VOC2'] = voc_testset.VOC2.str.replace('\n', ' ')

        voc = pd.concat([voc_testset.VOC1, voc_testset.VOC2]).sort_index().values
        voc_testset = pd.concat([voc_testset] * 2).sort_index().iloc[:, :-2]
        voc_testset['VOC'] = voc
        voc_testset['VOC'].fillna('', inplace=True)
        voc_testset['VOC'] = voc_testset['VOC'].apply(str)
        voc_testset.reset_index(inplace=True)
        voc_testset['label'] = pd.DataFrame(np.zeros((16, voc_testset.shape[0])).T).astype(int).apply(
            list, axis=1)
        return voc_testset
