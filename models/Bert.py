# coding: UTF-8
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class Config(object):
    """Config for parameters"""
    def __init__(self, dataset_path):
        self.model_name = 'Bert_Classification_Model'
        self.train_path = dataset_path + '/data/train.csv'                                # train set
        self.dev_path = dataset_path + '/data/test.csv'                                   # val set
        self.test_path = dataset_path + '/data/test.csv'                                  # test set
        self.content_key = 'OriginalTweet'                                                # field for text contents
        self.label_key = 'Sentiment'                                                      # filed for label
        self.label_dict = {"Extremely Positive": 0, "Positive": 1, "Neutral": 2, "Negative": 3, "Extremely Negative": 4}  # dict foe label encoding
        self.class_list = range(len(self.label_dict))                                     # label list
        self.num_classes = len(self.class_list)  # number of labels                       # number of label classes
        self.save_path = dataset_path + '/saved_dict/' + self.model_name + '.ckpt'        # path for model storing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        # devices

        self.num_epochs = 10                                            # number of training epochs
        self.patience = 3                                               # patience for early stopping
        self.batch_size = 32                                            # size of batch
        self.pad_size = 64                                              # max sequence size
        self.learning_rate = 1e-5                                       # learning rate
        self.bert_path = "bert-base-uncased"                            # Bert model
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)  # tokenizer
        self.hidden_size = 768                                          # hidden size of Bert
        self.dropout_rate = 0.5                                         # dropout rate


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooler_output = self.bert(context, attention_mask=mask)
        pooled = self.dropout(pooler_output)
        logits = self.fc(pooled)
        return logits
