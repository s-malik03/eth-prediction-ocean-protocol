# eth-prediction-ocean-protocol

## LSTM
### Usage
```
usage: stock-prediction.py [-h] [-p] [-m MODEL_PATH] [-it INPUT_TRAIN_PATH]
                           [-ip INPUT_PREDICT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -p, --predict         Set to prediction mode
  -m MODEL_PATH, --model_path MODEL_PATH
                        Model path
  -it INPUT_TRAIN_PATH, --input_train_path INPUT_TRAIN_PATH
                        Input path for the data for training
  -ip INPUT_PREDICT_PATH, --input_predict_path INPUT_PREDICT_PATH
                        Input path for the data to predict

```
### Training
```
python stock-prediction.py
```

### Prediction
The output will be output to the eth_prediction.csv
```
python stock-prediction.py -p
```
