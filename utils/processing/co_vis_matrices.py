from pathlib import Path
from typing import List, Dict
import numpy as np
import gc
import cudf
from utils.data.load_data import read_file
from utils.prep import RecursiveNamespace


def clicks(df, part, size, type_weight):
    df = df.sort_values(['session','ts'],ascending=[True,False])
    # USE TAIL OF SESSION
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount()
    df = df.loc[df.n<30].drop('n',axis=1)
    # CREATE PAIRS
    df = df.merge(df, on='session')
    df = df.loc[(df.aid_x != df.aid_y)]
    df = df.loc[(df.ts_x - df.ts_y).abs() < 1 * 24 * 60 * 60]
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= part * size) & (df.aid_x < (part + 1) * size)]
    # ASSIGN WEIGHTS
    df = (df[['session', 'aid_x', 'aid_y','ts_x']]
          .drop_duplicates(['session', 'aid_x', 'aid_y']))
    df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800) 
    return df


def cart_orders(df, part, size, type_weight):
    df = df.sort_values(['session','ts'],ascending=[True,False])
    # USE TAIL OF SESSION
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount()
    # df = df.loc[df.n<20].drop('n',axis=1)
    df = df.loc[df.n<30].drop('n',axis=1)
    # CREATE PAIRS
    df = df.merge(df, on='session')
    df = df.loc[(df.aid_x != df.aid_y)]
    df = df.loc[(df.ts_x - df.ts_y).abs() < 1 * 24 * 60 * 60]
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= part * size) & (df.aid_x < (part + 1) * size)]
    # ASSIGN WEIGHTS
    df = (df[['session', 'aid_x', 'aid_y','type_y']]
            .drop_duplicates(['session', 'aid_x', 'aid_y']))
    df['wgt'] = df.type_y.map(type_weight)  
    return df


def buy_to_buy(df, part, size, type_weight):
    # ONLY WANT CARTS AND ORDERS
    df = df.loc[df['type'].isin([1,2])] 
    df = df.sort_values(['session','ts'],ascending=[True,False])
    # USE TAIL OF SESSION
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount()
    # df = df.loc[df.n<100].drop('n', axis=1)
    df = df.loc[df.n<30].drop('n',axis=1)
    # CREATE PAIRS
    df = df.merge(df, on='session')
    # df = df.loc[(df.aid_x != df.aid_y)]
    df = df.loc[ ((df.ts_x - df.ts_y).abs()< 7 * 24 * 60 * 60) & (df.aid_x != df.aid_y) ] # 14 DAYS
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= part * size) & (df.aid_x < (part + 1) * size)]
    # ASSIGN WEIGHTS
    df = (df[['session', 'aid_x', 'aid_y','type_y']]
            .drop_duplicates(['session', 'aid_x', 'aid_y']))
    df['wgt'] = 1 
    # df['wgt'] = df.type_y.map({1: 1, 2: 1}) 
    return df


def cvm(cache_df: List, 
        files: List,
        tech: str,
        # tw_update: dict,
        *,
        SAVE_DIR: Path=Path('./output/co-vis-matrices'),
        ) -> None:
    
   
    # Save to Disk or Not
    file_exist = SAVE_DIR / f'top_{tech.top_n}_{tech.file_name}_v{tech.ver}_0.pqt'   
    if not file_exist.exists() or tech.overwrite:
        print(f'Creating *.pqt files for: {tech.fun_name}')
    else:
        print(f'Using Existing Files for: {file_exist}')
        return
    
    # Type weight
    type_weight = tech.type_weight
    if isinstance(type_weight, RecursiveNamespace):
        type_weight = type_weight.__dict__
        type_weight = dict([(int(key), value) for key, value in type_weight.items()])
        
    # CHUNK PARAMETERS
    if tech.file_name == 'buy2buy':
        DISK_PIECES = 1
        INNER_CHUNK_SIZE = 5
        AID_MAX_APPROX = 1.86E6
        OCS_DEN = 6
        OUTER_CHUNK_SIZE = int(np.ceil(len(files) / OCS_DEN))
    else:
        DISK_PIECES = 4
        INNER_CHUNK_SIZE = 3
        AID_MAX_APPROX = 1.86E6
        OCS_DEN = 8
        OUTER_CHUNK_SIZE = int(np.ceil(len(files) / OCS_DEN))
    print(f'We will process {len(files)} files:'
          f'\n\tOuter Chunks of {OUTER_CHUNK_SIZE} (i.e., Outer Chunks) and '
          f'{INNER_CHUNK_SIZE} (i.e., Inner Chunks)')
    
    # USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
    SIZE = AID_MAX_APPROX / DISK_PIECES

    # Top N to save for each session
    top_n = tech.top_n
    print(f'Starting: {tech.file_name}\n')
    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print('### DISK PART', PART + 1)
        
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(OCS_DEN):
            a = j * OUTER_CHUNK_SIZE
            b = min((j + 1) * OUTER_CHUNK_SIZE, len(files))
            print(f'Processing files {a} thru {b-1} in groups of {INNER_CHUNK_SIZE}...')
            
            # => INNER CHUNKS
            for k in range(a, b, INNER_CHUNK_SIZE):
                # READ FILE
                df = [read_file(cache_df=cache_df, f=files[k])]
                for i in range(1, INNER_CHUNK_SIZE): 
                    if k + i < b: df.append(read_file(cache_df=cache_df, f=files[k + i]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.sort_values(['session', 'ts'], ascending=[True, False])
                
                # Select Co-Vis- Matrix Operation
                cmv_fun = globals()[tech.fun_name]
                df = cmv_fun(df=df, part=PART, size=SIZE, type_weight=type_weight)
                
                df = df[['aid_x','aid_y','wgt']]
                df.wgt = df.wgt.astype('float32')
                df = df.groupby(['aid_x','aid_y']).wgt.sum()
                # COMBINE INNER CHUNKS
                if k==a: tmp2 = df
                else: tmp2 = tmp2.add(df, fill_value=0)
                print(k,', ',end='')
            print()
            # COMBINE OUTER CHUNKS
            if a==0: tmp = tmp2
            else: tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n<top_n].drop('n',axis=1)
        # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(SAVE_DIR / f'top_{top_n}_{tech.file_name}_v{tech.ver}_{PART}.pqt')
    return