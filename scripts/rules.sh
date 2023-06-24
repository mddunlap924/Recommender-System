#! /bin/bash

# Shell script to execute customized rules-only based model.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 ./src/rules_only.py --dir './cfgs' --name 'model-train-0val.yaml' &&
