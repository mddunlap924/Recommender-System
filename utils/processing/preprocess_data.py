"""
Data preprocessing
"""
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import time
from more_itertools import sliced
import os
import logging
import glob
import shutil

logger = logging.getLogger(__name__)
CHUNK_SIZE = 100_000


def real_sessions(df, time_gap: float=2.0):
    """Calculate the number of real sessions (rs) for each user (where a user is called a session)

    Args:
        df (_type_): Dataframe
        time_gap (float, optional): Time delta to classify a session [units=hours]. Defaults to 2.0.

    Returns:
        _type_: Dataframe with number of events and real sessions added to it
    """
    # Compute day and hour of activity
    df['day'] = df.ts.dt.day
    df['hour'] = df.ts.dt.hour
    
    # Number of times a session appears in data
    df = df.reset_index(drop=True) 
    df['num_events'] = df.groupby(['session']).aid.transform('count')
    
    # Number of "real users sessions" by looking at time gaps in activity
    df = df.sort_values(['session', 'ts'])
    df['rs'] = df.groupby('session').ts.diff()
    
    # Convert timedelta64[ns] into hours
    df['rs'] = (df.rs.dt.total_seconds() / 60) / 60
    
    # Calculate the number of real sessions by user
    df['rs'] = (df.rs > time_gap).astype('int8').fillna(0)
    df['rs'] = df.groupby('session').rs.cumsum()
    return df


def format_candidate(df):
    df = df.to_frame().reset_index()
    df = df.rename(columns={'index': 'session', 0: 'aid'})
    df = df.explode('aid').reset_index(drop=True)
    return df


def candidates_truth(df: pd.Series, tar: pd.DataFrame, matrix_type: str):
    """
    Combine the candidates and the labels
    """
    
    tar = tar.loc[tar['type']==matrix_type]
    aids = tar.ground_truth.explode().astype('int32').rename('aid')
    tar = tar[['session']].astype('int32')
    tar = tar.merge(aids, left_index=True, right_index=True, how='left')
    tar['target'] = 1
    
    comb = df.merge(tar, on=['session', 'aid'], how='left').fillna(0)

    return comb


def include_features(df: pd.DataFrame, feature: pd.DataFrame, merge_on: str):
    if isinstance(merge_on, list):
        df = df.merge(feature, on=merge_on, how='left').fillna(-1)
    else:
        df = df.merge(feature, on=[merge_on], how='left').fillna(-1)
    return df


def merge_candidates_features_dask_xgb(df, cfg, split_name, chunk_size, event):
    """
    Merge Candidates with Features for Model
    """
    ts = time.time()
    
    # Create split ranges
    df['session_count'] = (df.groupby('session').cumcount()==0).astype(int).cumsum()
    split_start = np.arange(1, df['session_count'].max(), chunk_size).tolist()
    split_end = [i + chunk_size - 1 for i in split_start]
    split_ranges = list(zip(split_start, split_end))
    
    # Create file save names
    save_dir = Path(f'./output/dmatrix/{split_name}')
        
    # Remove existing parquet and buffer files
    types = (str(save_dir / '*.parquet'), str(save_dir / '*.buffer'))
    current_files = []
    for files in types:
        current_files.extend(glob.glob(files))
    for current_file in current_files:
        os.remove(current_file)
    del current_files, files
    
    # Remove existing directories
    current_dirs = glob.glob(str(save_dir / '*'), recursive=True)
    for current_dir in current_dirs:
        shutil.rmtree(current_dir)
    del current_dirs
    
    # Convert Pandas DataFrame to Dask
    df = dd.from_pandas(df, npartitions=8)
    df['aid'] = df['aid'].astype(int)
    df['rec'] = df['rec'].astype(float)
    
    # Session, Aid, and Interaction Features
    print(f'\tLoading Features')
    logger.info(f'\tLoading Features')
    merge_ons, feature_paths = features_to_merge2(fbp=Path(cfg.path.features.base) / cfg.approach,
                                            event=event)
    
    # Merge Sessions and Aids into Clicks, Buys, Carts
    print(f'\tMerging Features')
    logger.info(f'\tMerging Features')
    for i, idx in enumerate(split_ranges):
        save_dir_folder = save_dir / f'split_{i+1}_of_{len(split_ranges)}'
        tmp = df[(df['session_count']>= idx[0]) & (df['session_count'] <= idx[1])]
        print(f'\ti={i}; idx_start={idx[0]:,}; idx_end={idx[1]:,}; len(dfc): {len(tmp):,}')
        for merge_on, feature_path in zip(merge_ons, feature_paths):
            feature = dd.read_parquet(feature_path)
            tmp = include_features(df=tmp, feature=feature, merge_on=merge_on)  
            # tmp['rec'] = tmp['rec'].astype(float)
            # tmp['cvm_weights'] = tmp['cvm_weights'].astype(float)
            tmp = tmp.astype(float)
            tmp['session'] = tmp['session'].astype(int)
            tmp['aid'] = tmp['aid'].astype(int)
            # print(f'\t\tmerged: {str(feature_path)}')
            # logger.info(f'\t\tmerged: {str(feature_path)}')
            del feature
            _ = gc.collect()     
            if i == 0:
                tmp.to_parquet(save_dir_folder)
            else:
                tmp.to_parquet(save_dir_folder)
            del tmp
            _ = gc.collect()
            tmp = dd.read_parquet(save_dir_folder)
    print(f'\tSaved Parquets at: {save_dir}')
    del df
    _ = gc.collect()

    mt = round((time.time() - ts) / 60, 3)
    print(f'Time to Merge Candidates + Features: {mt} mins.')
    return save_dir


def merge_candidates_features_dask_test(df, cfg, event):
    """
    Merge Candidates with Features for Model
    """
    ts = time.time()
    df = df.sort_values(by='session').reset_index(drop=True)
    print(f'In merge test; len(df): {len(df):,}')
    fbp = Path(cfg.path.features.base) / cfg.approach 
    
    # Session, Aid, and Interaction Features
    print(f'\tLoading Features')
    logger.info(f'\tLoading Features')
    merge_ons, features = features_to_merge(fbp=fbp)
    
    # Convert aid to int
    df['aid'] = df['aid'].astype(int)
    
    # Save model ready clicks, buys, carts to disk
    save_dir = Path(f'./output/model-input-data/{cfg.approach}')
    # Merge Sessions and Aids into Clicks, Buys, Carts
    print(f'\tMerging Features')
    logger.info(f'\tMerging Features')
    
    df['session_count'] = (df.groupby('session').cumcount()==0).astype(int).cumsum()
    chunk_size = 50_000
    split_start = np.arange(1, df['session_count'].max(), chunk_size).tolist()
    split_end = [i + chunk_size - 1 for i in split_start]
    split_ranges = list(zip(split_start, split_end))

    # Save model ready clicks, buys, carts to disk
    if event == 'orders':
        save_name = 'buys'
        save_dir = Path(f'./output/model-input-data/{cfg.approach}/{save_name}_temp')
    else:
        save_name = event
        save_dir = Path(f'./output/model-input-data/{cfg.approach}/{event}_temp')
    
    # Remove all files in Save Directory
    for file in os.scandir(save_dir):
        os.remove(file.path)
    
    save_paths = []
    print(f'len(df): {len(df):,}; df.session_count.max() = {df.session_count.max():,}')
    for i, idx in enumerate(split_ranges):
        dfc = df.copy()
        dfc = dfc[(dfc['session_count']>= idx[0]) & (dfc['session_count'] <= idx[1])]
        print(f'\ti={i}; idx_start={idx[0]:,}; idx_end={idx[1]:,}; len(dfc): {len(dfc):,}')
        
        dfc = dd.from_pandas(dfc, npartitions=8)
        for merge_on, feature in zip(merge_ons, features):
            dfc = include_features(df=dfc, feature=feature, merge_on=merge_on)          
        del merge_on, feature
        _ = gc.collect()
        
        # Convert Dask DataFrame to Pandas DataFrame
        dfc = dfc.compute()
        
        # Save model ready clicks, buys, carts to disk
        if event == 'orders':
            save_name = 'buys'
            save_dir = Path(f'./output/model-input-data/{cfg.approach}/{save_name}_temp')
        else:
            save_name = event
            save_dir = Path(f'./output/model-input-data/{cfg.approach}/{event}_temp')
        save_path_name = save_dir / f'{save_name}{i}.parquet'
        dfc.to_parquet(save_path_name)
        print(f'\tTest Candidate + Feature to Disk at: {str(save_path_name)}')
        logger.info(f'\tTest Candidate + Feature to Disk at: {str(save_path_name)}')
        save_paths.append(save_path_name)
        del dfc
        _ = gc.collect()
    del df
    return


def features_to_merge(fbp):
    dfs = dd.read_parquet(fbp / 'session' / 'basic.parquet')
       
    dfa = dd.read_parquet(fbp / 'aid' / 'basic.parquet')
    
    dfi = dd.read_parquet(fbp / 'interaction' / 'basic.parquet')
    
    print(f'\tCompleted Loading Features')
    logger.info(f'\tCompleted Loading Features')

    merge_ons = ['aid',
                'session',
                ['session', 'aid'],
                ]
    features = [dfa, dfs, dfi]
    del dfa, dfs, dfi
    _ = gc.collect()
    return merge_ons, features


def features_to_merge2(fbp, event):
    dfs = fbp / 'session' / 'basic.parquet'
    dfa = fbp / 'aid' / 'basic.parquet'
    dfi = fbp / 'interaction' / 'basic.parquet'
    print(f'\Identified Features')
    logger.info(f'\tIdentified Features')

    merge_ons = ['aid',
                'session',
                ['session', 'aid'],
                ]

    feature_paths = [dfa, dfs, dfi]
    del dfa, dfs, dfi
    
    _ = gc.collect()
    return merge_ons, feature_paths
    