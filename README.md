# Pretty-Pytorch-Text-Classification-Framework
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
        |   |__ train.csv --> Original training dataset
        |   |__ test.csv --> Original testing dataset
        |
        |__ saved_dict/ --> Model saving


|__model/
        |__ Bert/ --> Codes for Bert sequence classification model
        
|__ train_eval.py copied from huggingface/transformers
|__ read_data.py --> Codes for reading the dataset; forming labeled training set, unlabeled training set, development set and testing set; building dataloaders
|__ twitter_preprocesser.py --> Codes for BERT baseline model
|__ text_preprocesser.py --> Codes for training BERT baseline model
|__ utils.py --> Codes for our proposed TMix/MixText model
|__ run.py --> Codes for training/testing TMix/MixText 
