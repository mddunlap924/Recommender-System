import xgboost as xgb
from sklearn.model_selection import GroupKFold
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import logging
from utils.metrics import otto_metric_piece
import gc
import yaml
import os
import cudf
from utils.processing.preprocess_data import candidates_truth, merge_candidates_features_dask_xgb
from utils.processing.candidates import suggest_load_existing
import dask.dataframe as dd


logger = logging.getLogger(__name__)
FEATURE_COLS = []
CHUNK_SIZE = 25_000

# XGB use best iteration
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 # set iterator to 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = cudf.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target]) #, weight=dt['weight'])
        self.it += 1
        return 1


def stratify_folds(df):
    # Stratify Group K-Fold
    skf = GroupKFold(n_splits=5)

    strat = {}
    for fold,(train_idx, valid_idx) in enumerate(skf.split(df, df['target'], groups=df['session'])):
        strat[fold] = {'fold': fold,
                       'train_idx': train_idx,
                       'valid_idx': valid_idx,
                       'num_rows': len(df)}
    return strat



def train_xgb(df: pd.DataFrame, data_type: str, cfg, hps, cand_path: str) -> None:

    # Ground-truth labels
    test_labels = pd.read_parquet('./data/otto-validation/test_labels.parquet')
    # Combine candidates with labels
    print(f'Validation: Starting to Combine Canidates and Ground Truth - {data_type}')
    df = candidates_truth(df=df, tar=test_labels, matrix_type=data_type)
    print(f'\t Completed Combining Canidates and Ground Truth - {data_type}')
    
    # Get fold information
    strat = stratify_folds(df=df)
    del df
    _ = gc.collect()
    
    # Load features YAML and update global
    with open(hps.features, 'r') as file:
        features = yaml.safe_load(file)
    file.close()
    global FEATURE_COLS
    FEATURE_COLS = features
    del features
    print(f'Val. Training Features: {hps.features}')
  
    # Sessions from each fold prediction
    pred_sessions = {}
    
    # Column names for the Features
    feature_cols = FEATURE_COLS
    print(f'Feature Cols: {feature_cols}')
    logger.info(f'Feature Cols: {feature_cols}')
    for fold, fold_info in strat.items():
        print(f'\nFold {fold}: {data_type}')
        logger.info(f'\nFold {fold}: {data_type}')
        
        # Load Candidate Data to Disk
        df, cp = suggest_load_existing(bls=cfg.baseline_candidates,
                                       approach=cfg.approach,
                                       num_cand=cfg.candidates,
                                       event=data_type)
        if cp == cand_path:
            print('CP and cand_path match')
            logger.info('CP and cand_path match')
        else:
            print('Error need to fix cand_path')
            logger.info('Error need to fix cand_path')
        
        # Add ground-truth labels
        df = candidates_truth(df=df, tar=test_labels, matrix_type=data_type)
        
        # Add ranking weight for model predictor
        weights = np.logspace(1, 0.5, cfg.candidates, base=2, endpoint=True)-1
        df['cvm_weights'] = np.tile(weights, int(len(df) / cfg.candidates))


        # Split df into train and validation
        train = df.iloc[fold_info['train_idx']].sort_values(by=['session'], ascending=[True])
        train = train.reset_index(drop=True)
        val = df.iloc[fold_info['valid_idx']].sort_values(by=['session'], ascending=[True])
        val = val.reset_index(drop=True) 
        del df
        _ = gc.collect()   
        
        # Reduce train size
        print(f'Initial Train Len: {len(train):,}')
        logger.info(f'Initial Train Len: {len(train):,}')
        positives = train.loc[train['target']==1]
        negatives = train.loc[train['target']==0].sample(frac=hps.frac, random_state=42)
        train = pd.concat([positives,negatives], axis=0, ignore_index=True)
        del positives, negatives
        _ = gc.collect()
        train = train.sort_values(by=['session'], ascending=[True]).reset_index(drop=True)
        train_groups = train.session.value_counts()[train.session.unique()].values
        print(f'Downsamples Train Len: {len(train):,}')
        logger.info(f'Downsamples Train Len: {len(train):,}')
        
        
        # Merge Session and Aid features
        for field in ['session', 'aid', 'target']:
            train[field] = train[field].astype(int)
            val[field] = val[field].astype(int)
            
        print(f'XGB Merging Features for Train')
        logger.info(f'XGB Merging Features for Train')
        train_load_path = merge_candidates_features_dask_xgb(df=train, 
                                                             cfg=cfg,
                                                             split_name='train',
                                                             chunk_size=150_000,
                                                             event=data_type)
        del train
        _ = gc.collect()

        # VALIDATION DF and DMATRIX
        print(f'XGB Merging Features for Val.')
        logger.info(f'XGB Merging Features for Val.')
        print(f'Len. Val: {len(val):,}')
        val_len = len(val)
        val_load_path = merge_candidates_features_dask_xgb(df=val,
                                                           cfg=cfg,
                                                           split_name='validation',
                                                           chunk_size=40_000,
                                                           event=data_type)
        del val
        _ = gc.collect()
        
        # Val. dataframe to disk
        dmatrix_save_path = Path('./output/dmatrix/validation')

        # Reduce VAL size
        print(f'Initial Val. Len: {val_len:,}')
        logger.info(f'Initial Val. Len: {val_len:,}')
        val_file = sorted(glob.glob(str(val_load_path / '*')))[0]
        print(val_file)
        val = dd.read_parquet(val_file).compute()
        val = val[feature_cols + ['session', 'aid', 'target']]
        _ = gc.collect()
        val = val.sort_values(by=['session'], ascending=[True]).reset_index(drop=True)
        print(f'Downsamples Val. Len: {len(val):,}')
        logger.info(f'Downsamples Val. Len: {len(val):,}')
        val_groups = val.session.value_counts()[val.session.unique()].values
        
        # Create Val. DMATRIX
        dvalid = xgb.DMatrix(val[feature_cols], val['target'], group=val_groups)
        print(f'\tSave VALIDATION Dmatrix to Disk')
        dmatrix_val_path = dmatrix_save_path / 'validation.buffer'
        if dmatrix_val_path.exists():
            print(f'\tRemove DMatrix Validation currently at: {dmatrix_val_path}')
            os.remove(dmatrix_val_path)
        dvalid.save_binary(dmatrix_val_path)
        print(f'val. len: {len(val):,}')
        logger.info(f'val. len: {len(val):,}')
        del val, dvalid
        _ = gc.collect()
        
        print(f'\tSaved for Train and Val. DMatrix to Disk: {dmatrix_save_path}')

        # Save Path
        if data_type == 'orders':
            data_type = 'buys'
        save_path = Path('./output/models') / data_type / f'xgb_fold{fold}_{data_type}.xgb'
        
        # Load Data
        print(f'Reload Both Train Parquets and Val. Dmatrix from Disk')
        train = dd.read_parquet(train_load_path)
        train = train.sort_values(by=['session'], ascending=[True]).reset_index(drop=True)
        train = train.compute()
        print(f'\tLoaded Train DataFrame Parquet: {train_load_path}') 
        dvalid = xgb.DMatrix(dmatrix_val_path, nthread=-1)
        print(f'\tLoaded DMatrix Valid: {dmatrix_val_path}') 
        _ = gc.collect()
        
        # Train Iterative DataLoader
        Xy_train = IterLoadForDMatrix(train, feature_cols, 'target')
        dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256, nthread=-1)
        dtrain.set_group(train_groups)

        # Model Setup
        xgb_parms = {'objective':'rank:pairwise',
                     'tree_method':'gpu_hist',
                     'learning_rate': hps.learning_rate,
                     'max_depth': hps.max_depth,
                     'colsample_bytree': hps.colsample_bytree,
                     }
        model = xgb.train(xgb_parms, 
            dtrain=dtrain,
            evals=[(dtrain,'train'),
                   (dvalid,'valid')],
            num_boost_round=hps.num_boost_round,
            verbose_eval=hps.verbose_eval,
            early_stopping_rounds=hps.early_stopping_rounds,
            )
        del dtrain, dvalid, train, Xy_train
        _ = gc.collect()
        model.save_model(save_path)
        print(f'\tTrained and Saved: {save_path}')
        logger.info(f'\tTrained and Saved: {save_path}')
        # Store feature importance
        fp = dict(sorted(model.get_score(importance_type='gain').items(), key=lambda item: item[1], reverse=True))
        fp_print = []
        for key, value in fp.items(): fp_print.append(f'{key} = {value:.4f}')
        logger.info(fp_print)
        del model, fp, fp_print, key, value
        _ = gc.collect()
        
        # Evaluate on ValB to see how the model performed
        pred = val_inference(base_path=val_load_path, 
                             val_len=val_len,
                             data_type=data_type,
                             model_path=save_path)
        labels = test_labels[test_labels.session.isin(pred.index.unique())]
        print(f'\tLen. Pred and labels: {len(pred):,} & {len(labels):,}')
        logger.info(f'\tLen. Pred and labels: {len(pred):,} & {len(labels):,}')
        if data_type == 'buys':
            data_name = 'orders'
        else:
            data_name = data_type
        recall = otto_metric_piece(pred, data_name, labels, verbose=True)
        print(f'Recall {save_path}: {recall}')
        logger.info(f'Recall {save_path}: {recall}')
        pred_sessions[fold] = {'recall': recall}
        del pred, labels, recall
        _ = gc.collect()
        if data_type == 'buys':
            data_type = 'orders'
        
    return pred_sessions


def inference_single_model(df_path: Path, data_type: str, model_path: Path):
    df = dd.read_parquet(df_path).reset_index(drop=True).compute()
    print(f'df shape: {df.shape}')
    preds = np.zeros(len(df))
    # Column names for the Features
    model = xgb.Booster()
    model.load_model(model_path)
    model.set_param({'predictor': 'gpu_predictor'})
    dtest = xgb.DMatrix(data=df[FEATURE_COLS])
    # preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    preds = model.predict(dtest)
    predictions = df[['session','aid']].copy()
    predictions = predictions.reset_index(drop=True)
    predictions['pred'] = preds

    # predictions = predictions.sort_values(['session','pred'], ascending=[True,False]).reset_index(drop=True)
    print(f'predictions type: {type(predictions)}')
    print(f'predictions shape: {predictions.shape}')
    predictions = predictions.sort_values(['session','pred'], ascending=[True,False])
    predictions['n'] = predictions.groupby('session').aid.cumcount().astype('int8')
    predictions = predictions.loc[predictions.n<20]
    predictions = predictions.groupby('session').aid.apply(list)
    predictions.name = 'labels'
    
    return predictions


def val_inference(base_path, val_len, data_type, model_path):
    # data_paths = sorted(glob.glob(str(base_path / '*.parquet')))
    data_paths = sorted(glob.glob(str(base_path / '*')))
    data_paths = [i for i in data_paths if 'buffer' not in i]
    for ii, data_path in enumerate(data_paths):
        print(data_path)
        if ii == 0:
            preds = inference_single_model(df_path=data_path, data_type=data_type, model_path=model_path)
        else:
            pred_tmp = inference_single_model(df_path=data_path, data_type=data_type, model_path=model_path)
            preds = pd.concat([preds, pred_tmp])
    return preds


def inference(df: pd.DataFrame, data_type: str, features_path: str):
    
    if data_type == 'orders':
        data_type = 'buys'
    
    # Load features YAMLd and update global
    with open(features_path, 'r') as file:
        features = yaml.safe_load(file)
    file.close()
    
    # Column names for the Features
    preds = np.zeros(len(df))
    print(f'\tTest Inference Features: {features_path}')
    base_path = Path('./output/models') / data_type
    for fold in range(5):
        model = xgb.Booster()
        model.load_model(base_path / f'xgb_fold{fold}_{data_type}.xgb')
        model.set_param({'predictor': 'gpu_predictor'})
        dtest = xgb.DMatrix(data=df[features])
        preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1))/5
    predictions = df[['session','aid']].copy()
    predictions['pred'] = preds

    predictions = predictions.sort_values(['session','pred'],
                                          ascending=[True,False]).reset_index(drop=True)
    predictions['n'] = predictions.groupby('session').aid.cumcount().astype('int8')
    predictions = predictions.loc[predictions.n<20]
    predictions = predictions.groupby('session').aid.apply(list)
    predictions.name = 'labels'
    return predictions


def val_df_to_disk(df, base_save_path):
    current_parquets = glob.glob(str(base_save_path / '*.parquet'))
    for current_parquet in current_parquets:
        if 'train' not in current_parquet:
            os.remove(current_parquet)
        
    df['session_count'] = (df.groupby('session').cumcount()==0).astype(int).cumsum()
    chunk_size = CHUNK_SIZE
    split_start = np.arange(1, df['session_count'].max(), chunk_size).tolist()
    split_end = [i + chunk_size - 1 for i in split_start]
    split_ranges = list(zip(split_start, split_end))
    
    save_paths = []
    print(f'\tVal. to disk: len(df): {len(df):,}; df.session_count.max() = {df.session_count.max():,}')
    for i, idx in enumerate(split_ranges):

        dfc = df[(df['session_count']>= idx[0]) & (df['session_count'] <= idx[1])]
        print(f'\ti={i}; idx_start={idx[0]:,}; idx_end={idx[1]:,}; len(dfc): {len(dfc):,}')
        save_path_name = base_save_path / f'val_{i}.parquet'
        dfc.to_parquet(save_path_name)
        print(f'\tVal. DataFrame for XGB to Disk at: {str(save_path_name)}')
        logger.info(f'\tVal. DataFrame for XGB to Disk at: {str(save_path_name)}')
        save_paths.append(save_path_name)
    return save_paths
