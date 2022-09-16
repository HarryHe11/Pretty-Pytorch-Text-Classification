# coding: UTF-8
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class Config(object):
    """Config for parameters"""
    def __init__(self, dataset_path):
        self.model_name = 'Bert_Classification_Model'
        self.train_path = dataset_path + '/data/train.csv'                                # train set
        self.dev_path = dataset_path + '/data/dev.csv'                                    # val set
        self.test_path = dataset_path + '/data/test_set.xlsx'                                  # test set
        self.content_key = 'text'
        self.label_key = 'label'
        self.class_list = [0, 1, 2, 3]                                                       # label list
        self.num_classes = len(self.class_list)  # number of labels
        self.save_path = dataset_path + '/saved_dict/' + self.model_name + '.ckpt'        # path for model storing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        # devices

        self.num_epochs = 10                                            # number of training epochs
        self.patience = 5                                               # patience for early stopping
        self.batch_size = 16                                            # mini-batch
        self.pad_size = 64                                              # max sequence size
        self.learning_rate = 5e-5                                       # learning rate
        self.bert_path = "bert-base-uncased"                            # Bert model
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)  # tokenizer
        self.hidden_size = 768                                          # hidden size of Bert


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out
