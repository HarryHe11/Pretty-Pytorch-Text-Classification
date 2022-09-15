# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train_and_test
from utils import build_dataset, build_iterator, get_time_dif
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model in ../models', default = 'Bert')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Twitter_Stance'  # 数据集
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    seed = 1126
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure the results can be replicated

    start_time = time.time()
    print("Loading data...")
    test_data = build_dataset(config)
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    df_history = train_and_test(config, model, train_data, dev_data, test_iter)
    df_history.to_csv("training_results.csv",index=None)

