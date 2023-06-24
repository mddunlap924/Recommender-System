import pandas as pd, numpy as np
import os, gc
from utils.data.load_data import cache_data_to_cpu
from utils.processing import co_vis_matrices
from dotenv import load_dotenv
from utils import prep
import argparse
from pathlib import Path
import time
import pickle
from utils.processing.candidates import suggest, suggest_load_existing
from utils.log import setup_logger, log_cfg
from utils.processing.preprocess_data import merge_candidates_features_dask_test
from utils.models import xgb_ranker as xgb_ranker
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger()

# Load Environment variables from .env file
load_dotenv()
EVENTS = ['orders', 'clicks', 'carts']


if __name__ == "__main__":
    start_time = time.time()
    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = prep.debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = './cfgs'
        args.name = 'model-train-0val.yaml'
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

    # Setup Logger
    date_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    run_file_name = f'./output/submissions/{CFG.approach}/sub_{date_time}'
    logger = setup_logger(save_path=run_file_name + '.log')
    logger.info(f'\nYAML File: {args.name}\n')
    log_cfg(prep.load_cfg(base_dir=args.dir, filename=args.name, as_namespace=False))
    
    # Set random seed on everything
    prep.seed_everything(seed=CFG.seed)
    
    # Data Paths dependent on Validation or Test (i.e., submission)
    if CFG.approach == 'validation':
        data_paths = CFG.path.val
        performance = {}
    else:
        data_paths = CFG.path.test 
    
    # CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
    any_cvm = []
    any_overwrite = []
    for matrix_name in ['cart_orders', 'buy_to_buy', 'click']:
        tech = getattr(CFG.co_vis_matrix, matrix_name)
        file_exist = Path(CFG.co_vis_matrix.save_path) / CFG.approach / \
                    f'top_{tech.top_n}_{tech.file_name}_v{tech.ver}_0.pqt'
        file_exist = file_exist 
        any_cvm.append(file_exist.exists())
        any_overwrite.append(tech.overwrite)

    if (False in any_cvm) or (True in any_overwrite):
        # Cache data to RAM
        data_cache, files = cache_data_to_cpu(data_path=data_paths.base)
    
        # Co-Visitation Matrix: Cart-Orders
        co_vis_matrices.cvm(cache_df=data_cache,
                            files=files,
                            tech=CFG.co_vis_matrix.cart_orders,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)
        
        # Co-Visitation Matrix: Buy2Buy
        co_vis_matrices.cvm(cache_df=data_cache,
                            files=files,
                            tech=CFG.co_vis_matrix.buy_to_buy,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)
        
        # Co-Visitation Matrix: Clicks
        co_vis_matrices.cvm(cache_df=data_cache,
                            files=files,
                            tech=CFG.co_vis_matrix.click,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)

        # Memory management
        del data_cache, files
        _ = gc.collect()

    for event in EVENTS:
        print(f'\nStart Event: {event.upper()}')
        # Candidates
        if CFG.baseline_candidates.load:
            st = time.time()
            load_path = Path(CFG.baseline_candidates.path) / CFG.approach
            print(f'Starting to Load Data from: {load_path}')
            data, cand_path = suggest_load_existing(bls=CFG.baseline_candidates,
                                            approach=CFG.approach,
                                            num_cand=CFG.candidates,
                                            event=event)
            tt = round((time.time() - st) / 60, 3)
            print(f'Loaded Data from: {load_path}\n\t{tt} mins.')
            logger.info(f'Loaded Candidate Data from: {load_path}\n\t{tt} mins.')
            del load_path, st, tt
        else:
            print(f'Starting to Create Candidates for: {event}')
            
            data, cand_path = suggest(cfg=CFG,
                                      data_path=data_paths,
                                      load_path=Path(CFG.co_vis_matrix.save_path) / CFG.approach,
                                      event=event)

            print('Created Candidate Data - saved at: '
                f'{Path(CFG.co_vis_matrix.save_path) / CFG.approach}')
            logger.info('Created Candidate Data - saved at: '
                        f'{Path(CFG.co_vis_matrix.save_path) / CFG.approach}')
        
        # If Validation Train Model
        if CFG.approach == 'test':
            # Combine candidates with labels
            print(f'Test: Formatting Canidates Dataframe')
            # data = format_candidate(df=data)
            print(f'\t Completed Formatting Canidates Dataframe')
            check_cands = all(data.groupby('session')['aid'].count() == CFG.candidates)
            print(f'All Test Session Equal Number of Candidates: {check_cands}')
            logger.info(f'All Test Session Equal Number of Candidates: {check_cands}')
            # Add ranking weight for model predictor
            weights = np.logspace(1, 0.5, CFG.candidates, base=2, endpoint=True)-1
            data['cvm_weights'] = np.tile(weights, int(len(data) / CFG.candidates))
                
            # Merge Session and Aid features
            merge_candidates_features_dask_test(df=data, cfg=CFG, event=event) 
            _ = gc.collect()
            print(f'Test data created for {event.upper()}')
            logger.info(f'Test data created for {event.upper()}')

        # MODEL TRAINING
        if CFG.train_model and CFG.approach == 'validation':
            print(f'Starting Training')
            ts = time.time()
            data_train = xgb_ranker.train_xgb(df=data,
                                              data_type=event,
                                              cfg=CFG,
                                              hps=CFG.xgb,
                                              cand_path=cand_path)
            print(f'Trained: {event} in: {round((time.time() - ts) / 60, 3)} mins')
            logger.info(f'Trained: {event} in: {round((time.time() - ts) / 60, 3)} mins')
            
            # Log Avg. Recall for Each fold
            data_recall = np.array([value['recall'] for _, value in data_train.items()])
            performance[event] = {'mean': data_recall.mean(),
                                  'std': data_recall.std(),
                                  'info': data_train,
                                  'oof': pd.Series}
            print(f'{event} Recall Training: {data_recall.mean():.5f} ({data_recall.std():.3f})')
            logger.info(f'{event} Recall Training: {data_recall.mean():.5f} ({data_recall.std():.3f})')
            del data_train, data_recall
            _ = gc.collect()
            print(f'\nCompleted Event: {event.upper()}\n')
            # sys.exit()
        
    # Log Avg. Recall for Each fold
    if CFG.approach == 'validation':
        for event in EVENTS:
            print(f'{event} Recall Training: '
                    f'{performance[event]["mean"]:.5f} ({performance[event]["std"]:.3f})')
            logger.info(f'{event} Recall Training: '
                        f'{performance[event]["mean"]:.5f} ({performance[event]["std"]:.3f})')
        # Overall Recall from Training
        overall_recall = 0.1 * performance['clicks']['mean'] + \
                         0.3 * performance['carts']['mean'] + \
                         0.6 * performance['orders']['mean']
        print(f'Overall Recall: {overall_recall:.5f}')
        logger.info(f'Overall Recall: {overall_recall:.5f}')

    # Test Inference
    if CFG.approach == 'test':
        ts = time.time()
        pred = {}
        print('Starting Test Inference')
        for event in EVENTS:
            print(f'\nTest Inference for even: {event}')
            logger.info(f'\nTest Inference for even: {event}')
            if event == 'orders':
                save_name = 'buys'
            else:
                save_name = event
            
            save_dir = Path(f'./output/model-input-data/{CFG.approach}/{save_name}_temp')
            test_paths = []
            for test_path in  os.scandir(save_dir):
                test_paths.append(test_path)
            
            for ii, test_path in enumerate(test_paths):
                data = pd.read_parquet(test_path)
                
                print(f'\tStarting Test Inference for: {str(test_path)}; {ii + 1} of {len(test_paths)}')
                logger.info(f'\tStarting Test Inference for: {str(test_path)}; {ii + 1} of {len(test_paths)}')
                if ii == 0:
                    preds = xgb_ranker.inference(df=data, 
                                                    data_type=event,
                                                    features_path=CFG.xgb.features)
                else:
                    preds_tmp = xgb_ranker.inference(df=data, 
                                    data_type=event,
                                    features_path=CFG.xgb.features)
                    preds = pd.concat([preds, preds_tmp])
                    del preds_tmp
                print(f'\tCompleted Inference - {str(test_path)}')
                logger.info(f'\tCompleted Inference - {str(test_path)}')
            pred[event] = preds
            print(f'Completed Test Inference for event: {event}')
            logger.info(f'Completed Test Inference for event: {event}')
            del preds
        print(f'Completed Test Inference\n')
        logger.info(f'Completed Test Inference\n')

        # Create Submission File
        clicks_pred_df = pd.DataFrame(pred['clicks'].add_suffix("_clicks"),
                                    columns=["labels"]).reset_index()
        carts_pred_df = pd.DataFrame(pred['carts'].add_suffix("_carts"),
                                    columns=["labels"]).reset_index()
        orders_pred_df = pd.DataFrame(pred['orders'].add_suffix("_orders"),
                                    columns=["labels"]).reset_index()
        pred_df = pd.concat([clicks_pred_df, carts_pred_df, orders_pred_df,])
        pred_df.columns = ["session_type", "labels"]
        pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str,x)))
        date_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        print(f'Test Sub. Len: {len(pred_df):,}')
        logger.info(f'Test Sub. Len: {len(pred_df):,}')
        pred_df.to_csv(f'{run_file_name}.csv', index=False)

print(f'Script Execution: {round((time.time() - start_time) / 60, 2)} mins')
logger.info(f'Script Execution: {round((time.time() - start_time) / 60, 2)} mins')
print('End of Script\n')
