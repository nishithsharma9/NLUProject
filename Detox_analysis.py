import torch
import argparse
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset

from transformers import TrainingArguments,Trainer
import numpy as np
from torch import nn
from sklearn.utils import class_weight


def encode_data(df, text_column, tokenizer, max_seq_length=128):
    tokenized = tokenizer(list(df[text_column]), truncation=True,padding="max_length", max_length=max_seq_length)

    input_ids = torch.LongTensor(tokenized["input_ids"])
    attention_mask = torch.LongTensor(tokenized["attention_mask"])
    return input_ids,attention_mask

def extract_labels(dataset):
    labels = list(dataset["Label"])
    return labels

class Dataset(Dataset):
    def __init__(self, dataframe, text_column, tokenizer, max_seq_length=256):
        self.encoded_data = encode_data(dataframe, text_column, tokenizer,max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        input_ids,attention_mask = self.encoded_data
        labels = self.label_list
        dict_i = {}
        dict_i["input_ids"] = input_ids[i]
        dict_i["attention_mask"] = attention_mask[i]
        dict_i["label"] = labels[i]
        return dict_i
    
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    dic = {}
    accuracy = accuracy_score(labels, preds)
    dic["eval_accuracy"] = accuracy
    return dic

def model_init():
    return RobertaForSequenceClassification.from_pretrained("roberta-base")

parser = argparse.ArgumentParser(
    description="Test toxic transformer"
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Directory with toxic dataset",
)

args = parser.parse_args()


print("Get Data")
data = pd.read_csv(args.data_dir)

#Analysis will be done with toxic records
data = data[data.Updated==1]

#use 1% while running locally
data = data.sample(frac =.01)

print("Split Data")
train, test = train_test_split(data, test_size=0.2,random_state=10)
train, val = train_test_split(train, test_size=0.25,random_state=10)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


#Need to handle class imbalance by overriding computer_loss function in trainer
w0_count = train[train.Label == 0].Label.count()
w1_count = train[train.Label == 1].Label.count()
w0 = w1_count/(w0_count + w1_count)
w1 = w0_count/(w0_count + w1_count)
weights = torch.FloatTensor([w0,w1])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


#Toggle this to train either toxic or detox datasets
train_toxic = True
if train_toxic:

    print("Training")
    training_args = TrainingArguments(num_train_epochs = 2,  per_device_train_batch_size = 8, output_dir="results/toxic")
    train_dataset_toxic = Dataset(train,"Text", tokenizer)
    val_dataset_toxic = Dataset(val,"Text", tokenizer)
    test_dataset_toxic = Dataset(test, "Text", tokenizer)

    trainer_toxic = CustomTrainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset_toxic,
        eval_dataset=val_dataset_toxic,
        model_init=model_init,   
        compute_metrics=compute_metrics,
    )

    trainer_toxic.train()

    print("Make Predictions")
    predictions = trainer_toxic.predict(test_dataset_toxic)
    predictions = np.argmax(predictions.predictions, axis=-1)
    test["Predictions_toxic"] = predictions
    
    #change output path to whatever you want
    output_path = "/scratch/skp327/nlu_final/results/analysis_results_toxic"
    test.to_csv(output_path)
    print("Done")
    
else:

    print("Training")
    training_args = TrainingArguments(num_train_epochs = 2,  per_device_train_batch_size = 8, output_dir="results/detox")
    train_dataset_detox = Dataset(train,"Text_detox", tokenizer)
    val_dataset_detox = Dataset(val,"Text_detox", tokenizer)
    test_dataset_detox = Dataset(test, "Text_detox", tokenizer)

    trainer_detox = CustomTrainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset_detox,
        eval_dataset=val_dataset_detox,
        model_init=model_init,   
        compute_metrics=compute_metrics,
    )

    trainer_detox.train()

    print("Make Predictions")
    predictions = trainer_detox.predict(test_dataset_detox)
    predictions = np.argmax(predictions.predictions, axis=-1)
    test["Predictions_toxic"] = predictions
    
     #change output path to whatever you want
    output_path = "/scratch/skp327/nlu_final/results/analysis_results_detox"
    test.to_csv(output_path)
    print("Done")
