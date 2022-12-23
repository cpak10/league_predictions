# League of Legends Match Prediction Model
This repository contains scripts to collect data to train and use a model to predict the outcome of 
professional League of Legends matches.

The packages needed to run these scripts can be found in the following file: requirements.txt

## Data
Data to train and predict matches are pulled from Oracle's Elixir (https://oracleselixir.com/). The
process of pulling down data from this source is in flux, but can be found here: https://oracleselixir.com/tools/downloads

There is no current method of automatically pulling data from this site. Thus, some of the processes in the script, 
including downloading and renaming the file, will need to be done manually on the local desktop.

## Order of Operations
* pull_data.py
* train_prediction_model.py
* test_model.py
* predict_win.py