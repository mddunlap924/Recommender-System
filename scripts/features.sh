#! /bin/bash

# Shell script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 ./src/create_features.py --dir './cfgs/features' --name 'features-0.yaml' &&
wait
sleep 10

