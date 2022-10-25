# Pretty-Pytorch-Text-Classification
A (very pretty) Text Classification Model Training/Validating/Testing Template using Pytorch 

## Getting Started
These instructions will get you running the codes.

### Requirements
* Python >= 3.6
* Pytorch >= 1.3.0
* transformers == 3.1.0 
* Pandas, Numpy, torchmetrics


### Code Structure
```
|__ dataset/
|       |__ data/ --> Datasets
|       |   |__ train.csv --> Original training dataset, downloaded in https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
|       |   |__ test.csv --> Original testing dataset, downloaded in https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
|       |
|       |__ saved_dict/ --> Model saving
|
|__models/
|       |__ Bert/ --> Codes for Bert sequence classification model
|        
|__ bert_optimizer.py --> codes copied from huggingface
|__ train_eval.py --> Codes for model training, evaluation, and testing
|__ twitter_preprocesser.py --> Codes copied from twitter_preprocesser for tweet cleaning
|__ text_cleaner.py --> Codes for text_cleaning
|__ utils.py --> Codes for data loading
|__ run.py --> Codes for running the framework
```

### Set up config
Please set up all paramerters in ./models/Bert.py to use the framework.

### Training BERT baseline model
Please run `run.py` to train the BERT Sequence Classification model:
```
python ./run.py --model Bert
```

### Illustration of outputs of this framework
```
Loading data...
100% 41151/41151 [00:33<00:00, 1229.71it/s]
100% 3798/3798 [00:03<00:00, 1212.63it/s]
100% 3798/3798 [00:03<00:00, 1002.41it/s]
100% 41151/41151 [00:33<00:00, 1243.86it/s]
100% 3798/3798 [00:03<00:00, 1100.85it/s]
100% 3798/3798 [00:03<00:00, 1191.41it/s]
Time usage: 0:01:21

================================================================================2022-09-20 07:16:37
Epoch 1 / 10

100% 322/322 [06:30<00:00,  1.21s/it, train_acc=0.473, train_f1=0.479, train_loss=1.22]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.624, val_f1=0.639, val_loss=0.952]
<<<<<< reach best val_loss : 0.9516714890797933 >>>>>>

================================================================================2022-09-20 07:23:23
Epoch 2 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.683, train_f1=0.691, train_loss=0.844]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.652, val_f1=0.665, val_loss=0.942]
<<<<<< reach best val_loss : 0.9419301271438598 >>>>>>

================================================================================2022-09-20 07:30:14
Epoch 3 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.738, train_f1=0.745, train_loss=0.733]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.676, val_f1=0.688, val_loss=0.891]
<<<<<< reach best val_loss : 0.8913541555404663 >>>>>>

================================================================================2022-09-20 07:37:06
Epoch 4 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.772, train_f1=0.778, train_loss=0.659]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.667, val_f1=0.681, val_loss=0.906]

================================================================================2022-09-20 07:43:57
Epoch 5 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.772, train_f1=0.778, train_loss=0.658]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.681, val_f1=0.694, val_loss=0.886]
<<<<<< reach best val_loss : 0.8857783754666646 >>>>>>

================================================================================2022-09-20 07:50:48
Epoch 6 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.79, train_f1=0.796, train_loss=0.618]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.686, val_f1=0.699, val_loss=0.887]

================================================================================2022-09-20 07:57:39
Epoch 7 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.793, train_f1=0.799, train_loss=0.612]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.684, val_f1=0.698, val_loss=0.893]

================================================================================2022-09-20 08:04:29
Epoch 8 / 10

100% 322/322 [06:36<00:00,  1.23s/it, train_acc=0.793, train_f1=0.798, train_loss=0.612]
100% 30/30 [00:13<00:00,  2.27it/s, val_acc=0.68, val_f1=0.694, val_loss=0.892]
<<<<<< val_loss without improvement in 3 epoch, early stopping >>>>>>
100% 30/30 [00:13<00:00,  2.27it/s, test_acc=0.681, test_f1=0.694, test_loss=0.886]
<<<<<< Test Result >>>>>>
- test_loss : 0.8857783754666646
- test_acc : 0.6807165145874023
- test_f1 : 0.6937578916549683
                    precision    recall  f1-score   support

Extremely Positive       0.73      0.75      0.74       599
          Positive       0.61      0.63      0.62       947
           Neutral       0.74      0.78      0.76       617
          Negative       0.65      0.62      0.63      1041
Extremely Negative       0.73      0.70      0.72       592

          accuracy                           0.68      3796
         macro avg       0.69      0.70      0.69      3796
      weighted avg       0.68      0.68      0.68      3796
```

