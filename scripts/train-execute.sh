#! /bin/bash

# Shell script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, 
# models, hyperparameters, etc.

python3 ./src/train.py --dir './cfgs' --name 'model-train-0val.yaml' &&
wait
sleep 10

python3 ./src/train.py --dir './cfgs' --name 'model-train-0test.yaml' &&
wait
sleep 10
