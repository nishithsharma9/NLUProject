import pytorch_lightning as pl
import torch
from torchmetrics.functional import auroc
from transformers import AdamW, AutoModelForMaskedLM, AutoConfig
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import pickle

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
        """VOC train set removes a subset to use for validation"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """VOC validation set removes a subset to use for validation"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """VOC test set"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class NLU_Toxic_classifier(pl.LightningModule):
    """
    Binary Classification model for NLU Sentiment Analysis
    """
    def __init__(self, n_classes: int = 2,
                 n_training_steps=None,
                 n_warmup_steps=None):
        """
        Args:
            :param n_classes: Number of classes to classify
            :param n_training_steps: Number of training steps, not used for inferencing
            :param n_warmup_steps: Number of Warmup steps: not used for inferencing
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=True)
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-base", config=self.config)
        self.model.resize_token_embeddings(self.config.vocab_size+1) #[TOXIC] token added

        self.classifier = nn.Linear(self.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.hidden_states[-1]
        pooled_output = self.classifier(self.dropout(self.activation(self.dense(last_hidden_state[:, 0]))))
        loss = 0
        if labels is not None:
            loss = self.criterion(pooled_output, labels)
        return loss, torch.sigmoid(pooled_output)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        # self.log("predic_loss", loss, prog_bar=True, logger=True)
        return loss, outputs

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(np.array(['0', '1'])): #self.COLUMN_LABEL
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))
