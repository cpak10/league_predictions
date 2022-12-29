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
There are currently two different models being trained to accomplish the same goal: a sequential and a random forest.
Both models are required for the predict_win.py script. Use the following ordering for deploying the models:

* pull_data.py
* train_prediction_model.py
* test_model.py
* train_random_forest.py
* predict_win.py