#!/bin/bash

# Shell script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

yourfilenames=`ls ./cfgs/*.yaml`
for eachfile in $yourfilenames
do
   echo "$(basename "$eachfile")"
   FILENAME="$(basename "$eachfile")"
   python ./src/train.py --dir ./cfgs/ --name $FILENAME
   # echo $FILENAME
done
