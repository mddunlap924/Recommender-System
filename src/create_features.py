import pandas as pd
import gc
from utils.data.load_data import cache_data_to_cpu
from dotenv import load_dotenv
from utils import prep
import argparse
from pathlib import Path
import time
from utils import aid_features as aid_features
from utils import session_features as session_features
from utils import interaction_features


# Load Environment variables from .env file
load_dotenv()

SAVE = False
type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}
NORM_FACTOR = 'T'
ALL_COLUMNS = {}
PARTIAL = False

if __name__ == "__main__":
    start_time = time.time()
    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = prep.debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = './cfgs/features'
        args.name = 'features-0.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                         description= arg_desc)
        parser.add_argument("-cfg_basepath",
                            "--dir",
                            required=True,
                            help = "Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename",
                            "--name",
                            required=True,
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(f'\nARGS:\n{args}\n')
    CFG = prep.load_cfg(base_dir=args.dir, filename=args.name)

    
    # Set random seed on everything
    prep.seed_everything(seed=CFG.seed)
    
    # Base save path
    save_base = Path(CFG.path.save_base)
    
    # Data Paths dependent on Validation or Test (i.e., submission)
    # for approach in ['validation', 'test']:
    for approach in ['test']:
        print(f'Start Features for: {approach}')
        if approach == 'validation':
            data_paths = CFG.path.val
        else:
            data_paths = CFG.path.test
            
        # Cache data to RAM
        train_cache, _ = cache_data_to_cpu(data_path=data_paths.base, data_seg='train')
        test_cache, _ = cache_data_to_cpu(data_path=data_paths.base, data_seg='test')
        
        # Go from dictionary to single dataframe
        train = pd.concat([df for _, df in train_cache.items()])
        test = pd.concat([df for _, df in test_cache.items()])        
        print(f'\tCreated train and test dataframes for: {approach}')
        
        # STEP 2: AID FEATURES (train + test)
        print(f'\tStart Aids')
        base = pd.concat([train, test], ignore_index=True, axis=0)
        del train, train_cache, test_cache
        _ = gc.collect()
        base = base.sort_values(by=['aid', 'ts'], ascending=[True, True])
        if PARTIAL:
            base = base.iloc[0:1000, :].copy()
        df = aid_features.aid_basic(base.copy(), last_week=False)
        df = df.merge(aid_features.aid_ts_diff(base.copy()), on='aid', how='left')
        df.drop_duplicates(subset='aid', inplace=True)
        df = df.merge(aid_features.aid_lw_basic(base[base.ts >= test.ts.min()].copy()),
                      on='aid',
                      how='left').fillna(-1.0)
        df.drop_duplicates(subset='aid', inplace=True)
        df.reset_index(drop=True, inplace=True) 
        # Save to disk
        df.to_parquet(save_base / approach / 'aid' / 'basic.parquet')
        ALL_COLUMNS['aids'] = [col for col in df.columns.tolist() if col != 'aid']
        del df
        print(f'\tEnd Aids')
        
        # STEP 3: SESSION FEATURES (only test data)
        print(f'\tStart Session')
        base = test.copy()
        base = base.sort_values(by=['session', 'ts'], ascending=[True, True])
        if PARTIAL:
            base = base.iloc[0:1000, :].copy()
        df = session_features.session_basic(base)
        print('session 1')
        df = df.merge(session_features.session_ts_diff(base), on='session', how='left')
        print('session 2')
        df.drop_duplicates(subset='session', inplace=True)
        df.reset_index(drop=True, inplace=True) 
        # Save to disk
        df.to_parquet(save_base / approach / 'session' / 'basic.parquet')
        ALL_COLUMNS['sessions'] = [col for col in df.columns.tolist() if col != 'session']
        del df
        print(f'\tEnd Session')
      
        # Step 4: Interaction Features (only test data)  
        print(f'\tStart Interactions')
        base = test.copy()
        base = base.sort_values(by=['session', 'ts'], ascending=[True, True])
        if PARTIAL:
            base = base.iloc[0:1000, :].copy()
        df = interaction_features.inter_basic(base.copy()) 
        df = df.merge(interaction_features.inter_hour_day(base.copy()),
                      on=['session', 'aid'],
                      how='left')
        df.drop_duplicates(subset=['session', 'aid'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(save_base / approach / 'interaction' / 'basic.parquet')
        ALL_COLUMNS['inter'] = [i for i in df.columns.tolist() \
            if i != 'aid' if i != 'session']
        del df
        print(f'\tEnd Interactions')
        
        # PRINT COLUMN NAMES
        all_cols = []
        for key, val in ALL_COLUMNS.items():
            for val_ in val:
                all_cols.append(val_)
        print(f'All Columns: {len(all_cols)}')
        for feat in all_cols:
            print(f'- {feat}')
        
print('End of Script - Complete')
