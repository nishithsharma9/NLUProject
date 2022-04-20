import re
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import time
import datetime
import pickle
from collections import Counter
import pytorch_lightning as pl
from data import NLU_DataModule, VOC_Data_Transform
from model import VOC_TopicLabeler
import torch.nn as nn
from __init__ import cli_VOC_logo
import logging
import warnings
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class NLU_Detox:
    def __init__(self,
                 N_EPOCHS: int = 10,
                 BATCH_SIZE: int = 12,
                 MAX_LEN: int = 256,
                 LR: float = 2e-05,
                 opt_thresh: float = 0.4):
        """
        Args:
            :param N_EPOCHS: Number of Epochs
            :param BATCH_SIZE: Number of Batch Size
            :param MAX_LEN: Maximum length of Padding
            :param LR: Learning Rate
            :param opt_thresh: Optinmal Threshold for logit classification
        """
        
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)
        self.logger = logging.getLogger('sema_logger')

        self.N_EPOCHS = N_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LEN = MAX_LEN
        self.LR = LR
        self.opt_thresh = opt_thresh

        self.directory = os.path.abspath(os.getcwd())

        self.mlb = MultiLabelBinarizer()
        val['Label'] = mlb.fit_transform(val.Label.astype(str)).tolist()
        train['Label'] = mlb.fit_transform(train.Label.astype(str)).tolist()
        test['Label'] = mlb.fit_transform(test.Label.astype(str)).tolist()


        self.new_model = VOC_TopicLabeler.load_from_checkpoint(
            checkpoint_path=self.directory + "/model_weights/DeBERTa_220407_4.ckpt",
            n_classes=len(LABEL_COLUMNS),
            config=self.config)
        self.new_model.eval()

    def trainer_setup(self, voc_testset):
        """Setting up data module & trainer for inference
            Args:
                :param voc_testset: dataframe from inference dataset
            Returns:
                None: just update voc_testset
        """
        self.logger.info('Load Data Module & Trainer for inference...')
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        
        self.data_module = VOC_DataModule(self.voc_testset,
                                          self.voc_testset,
                                          tokenizer=self.tokenizer,
                                          batch_size=self.BATCH_SIZE,
                                          max_token_len=self.MAX_LEN)
        self.data_module.setup()
        self.trainer = pl.Trainer(max_epochs=self.N_EPOCHS, progress_bar_refresh_rate=1)
    
    def train(self):
        """Training the dataset"""
        self.logger.info('Training the dataset...')
        new_model = NLU_Toxic_classifier(n_classes=2,
                                         n_warmup_steps=warmup_steps,
                                         n_training_steps=total_training_steps)
        data_module = NLU_DataModule(train,
                                     val,
                                     tokenizer,
                                     batch_size=BATCH_SIZE,
                                     max_token_len=MAX_LEN)
        data_module.setup()
        trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, progress_bar_refresh_rate=3)
        trainer.fit(new_model, data_module)
        trainer.save_checkpoint(self.directory + "/model_weights/DeBERTa_220407_4.ckpt")
        
    def inference(self):
        """Inferencing the dataset"""
        self.logger.info('Inferencing the dataset...')
        trainer.predict(new_model, data_module)
        pred = pd.Series(prd).apply(lambda x : torch.argmax(x[1],axis=1).tolist()).apply(pd.Series).stack().reset_index(drop=True)
        del predicted
        
    def save_output(self, file: str = None):
        """Saving Outputs in Excel"""
        self.logger.info('Saving the Final Result...')
        self.voc_testset.fillna('').astype(str).to_excel(
            self.directory + '/output/' + file + '_output.xlsx',
            encoding='utf-8-sig', engine=engine)

    def process_analysis(self):
        "Run all the methods for all files"
        self.logger.info('Working on ' + file + '...')
        voc_testset = VOC_Data_Transform(file=file).transform()
        self.trainer_setup(voc_testset)
        self.inference()
        self.VOC_filter_etc()
        self.VOC_find_keyword()
        self.save_output(file=file)

        self.logger.info('DONE. Please check the output folder.')
        
if __name__ == "__main__":
    NLU_Detox().process_analysis()
