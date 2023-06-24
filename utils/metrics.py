import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# multiprocessing 
import psutil
N_CORES = psutil.cpu_count()     # Available CPU cores
print(f"N Cores : {N_CORES}")
from multiprocessing import Pool

weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
benchmark = {"clicks":0.5255597442145808, "carts":0.4093328152483512, 
             "orders":0.6487936598117477, "all":.5646320148830121}


def df_parallelize_run(func, t_split):
    
    num_cores = np.min([N_CORES, len(t_split)])
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    
    return df


def hits(b):
    # b[0] : session id
    # b[1] : ground truth
    # b[2] : aids prediction 
    return b[0], len(set(b[1]).intersection(set(b[2]))), np.clip(len(b[1]), 0, 20)


def otto_metric_piece(values, typ, valid_labels, verbose=True):
    c1 = pd.DataFrame(values, columns=["labels"]).reset_index().rename({"index":"session"}, axis=1)
    a = valid_labels.loc[valid_labels['type']==typ].merge(c1, how='left', on=['session'])

    b=[[a0, a1, a2] for a0, a1, a2 in zip(a["session"], a["ground_truth"], a["labels"])]
    c = df_parallelize_run(hits, b)
    c = np.array(c)
    
    recall = c[:,1].sum() / c[:,2].sum()
    
    print('{} recall = {:.5f} (vs {:.5f} in benchmark)'.format(typ ,recall, benchmark[typ]))
    logger.info(f'{typ} Recall = {recall:.5f}')
    return recall


def otto_metric(clicks, carts, orders, valid_labels, *, verbose=True):
    
    score = 0
    score += weights["clicks"] * otto_metric_piece(clicks, "clicks", valid_labels, verbose=verbose)
    score += weights["carts"] * otto_metric_piece(carts, "carts", valid_labels, verbose=verbose)
    score += weights["orders"] * otto_metric_piece(orders, "orders", valid_labels, verbose=verbose)
    
    if verbose:
        print('=============')
        print('Overall Recall = {:.5f} (vs {:.5f} in benchmark)'.format(score, benchmark["all"]))
        print('=============')
        logger.info(f'Overall Recall = {score:.5f}')
    return score