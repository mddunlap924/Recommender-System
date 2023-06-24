import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import cudf, itertools
from utils.data.load_data import cache_data_to_cpu, load_cvm, load_test
from utils.processing import co_vis_matrices
from dotenv import load_dotenv
from utils import prep
import argparse
from pathlib import Path
import time
import pickle
from utils.processing.candidates11 import suggest, suggest_load_existing
from utils.metrics import otto_metric, otto_metric_piece
from utils.log import setup_logger, log_cfg
from datetime import datetime
from utils.prep import RecursiveNamespace

import logging
logger = logging.getLogger()


# Load Environment variables from .env file
load_dotenv()


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
                            # tw_update=CFG.type_weight.__dict__,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)
        
        # Co-Visitation Matrix: Buy2Buy
        co_vis_matrices.cvm(cache_df=data_cache,
                            files=files,
                            tech=CFG.co_vis_matrix.buy_to_buy,
                            # tw_update=CFG.type_weight.__dict__,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)
        
        # Co-Visitation Matrix: Clicks
        co_vis_matrices.cvm(cache_df=data_cache,
                            files=files,
                            tech=CFG.co_vis_matrix.click,
                            # tw_update=CFG.type_weight.__dict__,
                            SAVE_DIR=Path(CFG.co_vis_matrix.save_path) / CFG.approach)

        # Memory management
        del data_cache, files
        _ = gc.collect()
    
    # Suggestions
    if CFG.baseline_candidates.load:
        st = time.time()
        load_path = Path(CFG.baseline_candidates.path) / CFG.approach
        print(f'Starting to Load Data from: {load_path}')
        (val_clicks,
         val_buys,
         val_carts) = suggest_load_existing(bls=CFG.baseline_candidates,
                                            approach=CFG.approach,
                                            num_cand=CFG.candidates)
        tt = round((time.time() - st) / 60, 3)
        print(f'Loaded Data from: {load_path}\n\t{tt} mins.')
        # logger.info('Partial Load of Candiate Data (for debugging)')
        logger.info(f'Loaded Candidate Data from: {load_path}\n\t{tt} mins.')
        del load_path, st, tt
    else:
        print(f'Starting to Create Candidates')
        # (val_clicks,
        #  val_buys,
        #  val_carts) = suggest(cfg=CFG,
        #                       data_path=data_paths,
        #                       load_path=Path(CFG.co_vis_matrix.save_path) / CFG.approach)
        # val_clicks, _ = suggest(cfg=CFG,
        #                 data_path=data_paths,
        #                 load_path=Path(CFG.co_vis_matrix.save_path) / CFG.approach,
        #                 event='clicks')
        # val_carts, _ = suggest(cfg=CFG,
        #         data_path=data_paths,
        #         load_path=Path(CFG.co_vis_matrix.save_path) / CFG.approach,
        #         event='carts')
        val_buys, _ = suggest(cfg=CFG,
                        data_path=data_paths,
                        load_path=Path(CFG.co_vis_matrix.save_path) / CFG.approach,
                        event='orders')
        print('Created Candidate Data - saved at: '
              f'{Path(CFG.co_vis_matrix.save_path) / CFG.approach}')
        logger.info('Created Candidate Data - saved at: '
                    f'{Path(CFG.co_vis_matrix.save_path) / CFG.approach}')
    
    # If Validation Make Predictions and Get Recall Scores
    if CFG.approach == 'validation':
        val_labels = pd.read_parquet(data_paths.test_labels)
        
        # Recall Metrics
        recall = otto_metric_piece(val_buys, 'orders', val_labels, verbose=True)
        # scores = otto_metric(clicks=val_clicks,
        #                      carts=val_carts,
        #                      orders=val_buys,
        #                      valid_labels=val_labels.copy()
        #                     )
    
    # Create Submission File
    clicks_pred_df = pd.DataFrame(val_clicks.add_suffix("_clicks"),
                                  columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(val_buys.add_suffix("_orders"),
                                  columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(val_carts.add_suffix("_carts"),
                                 columns=["labels"]).reset_index()
    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str,x)))
    date_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    if CFG.approach == 'test':
        pred_df.to_csv(f'{run_file_name}.csv', index=False)
    
    print(f'Script Execution: {round((time.time() - start_time) / 60, 2)} mins')
    logger.info(f'Script Execution: {round((time.time() - start_time) / 60, 2)} mins')
    print('End of Script\n')

        