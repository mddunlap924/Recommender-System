import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import cudf, itertools
from utils.data.load_data import cache_data_to_cpu, load_cvm, load_test
from utils import prep
import argparse
from pathlib import Path
import time
import pickle
from utils import aid_features
from datetime import timedelta


type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}


def aid_basic(df, *, last_week=False):
    df['tsd'] = pd.to_datetime(df['ts'], unit='s')
    df['day'] = df.tsd.dt.dayofweek
    df['hour'] = df.tsd.dt.hour
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    if last_week:
        dfc = df.groupby('aid').agg({'session': ['count', 'nunique'],
                    'type': ['mean', 'min', 'max'],
                    'ts': ['min', 'max', 'mean', 'std'],
                    'day': ['mean', 'max'],
                    'hour': ['mean', 'max'],
                    }).fillna(-1.0)
    else:
        dfc = df.groupby('aid').agg({'session': ['count', 'nunique'],
                                    'type': ['mean', 'min', 'max'],
                                    'day': ['mean', 'max'],
                                    'hour': ['mean', 'max'],
                                    }).fillna(0.0)

    dfc.columns = ['aid_' + '_'.join(col).strip() for col in dfc.columns.values]
    dfc.reset_index(inplace=True)
    dfc.drop_duplicates(subset='aid', inplace=True)
    print(f'\taid1') 
    
    # Interaction by day-of-week
    df0 = df['aid'].copy()
    df0 = df0.to_frame().drop_duplicates(subset='aid')
    day_col_names = []
    for day in range(7):
        day_col_name = f'aid_inter_day_{day}'
        day_col_names.append(day_col_name)
        tmp = df[df['day'] == day][['session', 'aid']].copy().reset_index(drop=True)
        tmp[day_col_name] = tmp.groupby('aid')['session'].transform('count')
        tmp = tmp[['aid', day_col_name]].drop_duplicates(subset='aid')
        df0 = df0.merge(tmp, on='aid', how='left').fillna(0)
        del tmp
        _ = gc.collect()
    df0 = df0[['aid'] + day_col_names]
    df0 = df0.drop_duplicates(subset='aid')
    print(f'\tmerge aid day')
    dfc = dfc.merge(df0, on='aid', how='left')
    print(f'\t\tcompleted aid session day')
    del df0
    _ = gc.collect()
    
    # Percent count of clicks, carts, and orders by aid
    df0 = df.copy()
    df0['aid_count'] = df0.groupby(['aid', 'type'])['aid'].transform('count')
    df0.reset_index(drop=True, inplace=True)
    cols = []
    for event, label in type_labels.items():
        col = f'aid_percent_count_{event}'
        cols.append(col)
        tmp = df0[df0.type == label].groupby(['aid'])['aid'].count()
        tmp.name = col             
        df0 = df0.merge(tmp, on='aid', how='left')
        df0.drop_duplicates(subset='aid', inplace=True)
    print(f'\taid2')
            
    for col in cols:
        df0[col] = df0[col] / df0['aid_count']
    df0.fillna(0.0, inplace=True)
    df0.drop_duplicates(subset='aid', inplace=True)
    df0 = df0[['aid'] + cols]
    print(f'\taid3')
    
    dfc = dfc.merge(df0, on='aid', how='left')
    dfc.drop_duplicates(subset='aid', inplace=True)
    del cols, df0
    print(f'\taid4')
    
    # Number of unique sessions for clicks, carted, and ordered an aid
    cols = []
    for event, label in type_labels.items():
        tmp = df[df.type == label].groupby('aid').agg({'session': ['count', 'nunique']})
        tmp.columns = [f'aid_' + '_'.join(col).strip() + f'_{event}' for col in tmp.columns.values]
        for col in tmp.columns.tolist():
            cols.append(col)
        tmp = tmp.reset_index()
        tmp = tmp.drop_duplicates(subset='aid')
        dfc = dfc.merge(tmp, on='aid', how='left').fillna(0.0)
        dfc.drop_duplicates(subset='aid', inplace=True)
        del tmp
        
    # Normalize counts by days
    cols += ['aid_session_count', 'aid_session_nunique']
    cols += day_col_names
    for col in cols:
        # if 'nunique' not in col:
        dfc[col] = dfc[col] / T
    print(f'\taid5')
    return dfc


def aid_ts_diff(df):
    Tmin = df.ts.min()
    df.ts -= Tmin
    grp_by = ['aid', 'session']
    df['aid_ts_diff'] = df.groupby(grp_by)['ts'].transform('diff')
    df['aid_ts_diff_avg'] = df['aid_ts_diff'] / df.groupby(grp_by)['ts'].transform('mean')
    df['aid_ts_diff_min'] = df['aid_ts_diff'] / df.groupby(grp_by)['ts'].transform('min')
    df['aid_ts_diff_max'] = df['aid_ts_diff'] / df.groupby(grp_by)['ts'].transform('max')
    df.fillna(0, inplace=True)
    df = df[['aid', 'aid_ts_diff_avg', 'aid_ts_diff_min', 'aid_ts_diff_max']]
    df = df.drop_duplicates(subset='aid')
    return df


def aid_lw_ts(df, date_cut):
    df = df[df.ts >= date_cut]
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    df = df.groupby('aid').agg({'ts': ['mean', 'std', 'min', 'max']})
    df.columns = ['aid_LW_' + '_'.join(col).strip() for col in df.columns.values]
    return df


def aid_lw_basic(df):
    df = aid_basic(df, last_week=True)
    df_cols_rename = {}
    for col in df.columns.tolist():
        if col != 'aid':
            # print(col)
            df_cols_rename[col] = 'LW_' + col
    df.rename(columns=df_cols_rename, inplace=True)
    
    return df



