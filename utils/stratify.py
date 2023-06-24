
from sklearn.model_selection import GroupKFold
from pathlib import Path
import pandas as pd
import numpy as np




def stratify_folds(df):
    
    # Test labels for evaluation
    test_labels = pd.read_parquet('./data/otto-validation/test_labels.parquet')
    
    skf = GroupKFold(n_splits=5)

    strat = {}
    for fold,(train_idx, valid_idx) in enumerate(skf.split(df, df['target'], groups=df['session'])):
        strat[fold] = {'fold': fold,
                       'train_idx': train_idx,
                       'valid_idx': valid_idx,
                       'num_rows': len(df)}
    return strat