# Pretty-Pytorch-Text-Classification
A (very pretty) Text Classification Framework using Pytorch 

## Getting Started
These instructions will get you running the codes.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* transformers == 3.1.0 
* Pandas, Numpy, torchmetrics


### Code Structure
```
|__ dataset/
        |__ data/ --> Datasets
        |   |__ train.csv --> Original training dataset, downloaded in https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
        |   |__ test.csv --> Original testing dataset, downloaded in https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
        |
        |__ saved_dict/ --> Model saving


|__models/
        |__ Bert/ --> Codes for Bert sequence classification model
        
|__ bert_optimizer.py --> codes copied from huggingface
|__ train_eval.py --> Codes for model training, evaluation, and testing
|__ twitter_preprocesser.py --> Codes copied from twitter_preprocesser for tweet cleaning
|__ text_cleaner.py --> Codes for text_cleaning
|__ utils.py --> Codes for data loading
|__ run.py --> Codes for running the framework
```

#### Set up config
Please set up all paramerters in ./models/Bert.py to use the framework.

#### Training BERT baseline model
Please run `run.py` to train the BERT Sequence Classification model:
```
python .run.py --model Bert
```

