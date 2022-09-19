# coding: UTF-8
import sys
import time
import datetime
from copy import deepcopy


from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn import metrics
from torchmetrics import Accuracy, F1Score


from utils import get_time_dif
from bert_optimizer import BertAdam


def test_model(config, model, loss_fn, metrics_dict, test_data):
    # 3，test -------------------------------------------------
    model.load_state_dict(torch.load(config.save_path))
    test_step_runner = StepRunner(model = model, stage = "test",
                                 loss_fn = loss_fn, metrics_dict = deepcopy(metrics_dict))
    test_epoch_runner = EpochRunner(test_step_runner)
    with torch.no_grad():
        test_metrics = test_epoch_runner(test_data)
    print("<<<<<< Test Result >>>>>>")
    for name, metric in test_metrics.items():
        print("- {0} : {1}".format(name, metric))

# api for training and testing models
def train_and_test(config, model, train_iter, dev_iter, test_iter):
    loss_fn = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())  # get all parameters
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = config.learning_rate,
                         warmup = 0.05,
                         t_total = len(train_iter) * config.num_epochs)
    metrics_dict = {"acc": Accuracy().to(config.device),'f1': F1Score().to(config.device)}
    df_history = train_model(config, model, optimizer, loss_fn, metrics_dict,
                             train_data = train_iter, val_data = dev_iter, monitor="val_f1")
    test_model(config, model, loss_fn, metrics_dict, test_iter)
    return df_history


def print_log(info):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(str(info) + "\n")


class StepRunner:
    '''Step runner for each training steps.
    Param:
        model: the training model
        loss_fn: loss function
        stage: current stage of the model. Default: 'train'.
        metric_dict: a dictionary for all selected metrics
        optimizer: the selected optimizer for model training
    '''
    def __init__(self, model, loss_fn, stage="train", metrics_dict=None, optimizer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.stage = stage
        self.metrics_dict = metrics_dict
        self.optimizer = optimizer

    def step(self, features, labels):
        # loss
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.model.train()  # training mode, the dropout layer works
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.model.eval()  # eval mode, the dropout layer doesn't work
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    '''Step runner for each training epoch.
    Param:
        step_runner: the step_runner for each training step
    '''
    def __init__(self, step_runner):
        self.step_runner = step_runner
        self.stage = self.step_runner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics = self.step_runner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.step_runner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name, metric_fn in self.step_runner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(config, model, optimizer, loss_fn, metrics_dict,
                train_data, val_data=None, monitor="val_loss"):
    epochs = config.num_epochs
    ckpt_path = config.save_path
    patience = config.patience
    history = {}
    for epoch in range(1, epochs + 1):
        print_log("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(model = model, stage = "train",
                                       loss_fn = loss_fn, metrics_dict = deepcopy(metrics_dict),
                                       optimizer = optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(model = model, stage = "val",
                                         loss_fn = loss_fn, metrics_dict = deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(model.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor, arr_scores[best_score_idx]))
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience))
            break
        model.load_state_dict(torch.load(ckpt_path))
    return pd.DataFrame(history)





