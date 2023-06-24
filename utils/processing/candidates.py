import pandas as pd
import dask.dataframe as dd
import itertools
import numpy as np
from collections import Counter
import pickle
import time
from utils.data.load_data import load_cvm, load_test, load_cvm_partial
from pathlib import Path
import gc

import logging
logger = logging.getLogger(__name__)

# multiprocessing 
import psutil
N_CORES = psutil.cpu_count()     # Available CPU cores
from multiprocessing import Pool

# Set some global variables that will be updated later
type_weight_multipliers = {}
top_20_clicks = {}
top_20_buy2buy = {}
top_20_buys = {}
top_clicks = pd.DataFrame
top_orders = pd.DataFrame
top_carts = pd.DataFrame
top_all = pd.DataFrame
nn_aids = np.array
w2v_aids = np.array
bpr_aids = np.array
bpr_sessions = np.array
CANDIDATES = int
CAND_FRAC = int
CLICK_LIM = int
B2B_LIM = int
BUYS_LIM = int

# Multi-argument Pooling (see example with 161 upvotes)
# https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python


def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES, len(t_split)])
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    return df


def suggest_clicks(df):
    session = df[0]
    aids = df[1]
    types = df[2]
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    # unique_carts_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))
    
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=CANDIDATES:
        weights=np.logspace(0.1, 1, len(aids), base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(CANDIDATES)]
        
        # max_count = 1.0 * aids_temp.most_common(1)[0][1]
        # for k in aids_temp: aids_temp[k] /= max_count
        max_count = aids_temp.most_common(1)[0][1]
        min_count = aids_temp.most_common(CANDIDATES)[-1][1]
        for k in aids_temp: aids_temp[k] = 1 + 0.5*(max_count - min_count)/(max_count-min_count)
        top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES)]
        final = [sorted_aids, top_aids_cnt]
        
        return session, final
    
    # USE "CLICKS" CO-VISITATION MATRIX
    aids_temp = Counter() 
    aids1 = list(itertools.chain(*[top_20_clicks[aid][:CLICK_LIM] for aid in unique_aids if aid in top_20_clicks]))
    for aid in aids1: aids_temp[aid] += 0.1
    
    aids2 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_aids if aid in top_20_buys]))
    for aid in aids2: aids_temp[aid] += 0.1
    
    aids3 = list(itertools.chain(*[w2v_aids[aid][:5] for aid in unique_aids]))
    for aid in aids3: aids_temp[aid] += 0.1
        
    # RERANK CANDIDATES
    top_aids2 = [aid for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    result = unique_aids + top_aids2[:CAND_FRAC - len(unique_aids)]
    result = result + [aid for aid in top_clicks if aid not in result]
    result = result[:CANDIDATES]
    if len(result) < CANDIDATES:
        print(f'{len(result)} < {CANDIDATES}: session {session}')
        logger.info(f'PROBLEM!!! CLICKS: {len(result)} < {CANDIDATES}: session {session}')
    
    max_count = 1.0 * aids_temp.most_common(1)[0][1]
    for k in aids_temp: aids_temp[k] /= max_count
    top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    rec_aids = [1.5 for i in range(len(unique_aids))] + top_aids_cnt
    rec_aids = rec_aids[:CANDIDATES] 
    if len(result) != len(rec_aids):
        rec_aids = (rec_aids + CANDIDATES * [0.0])[:CANDIDATES]
    if len(result) != len(rec_aids):
        print(f'PROBLEM!!! SUGGEST_CLICKS- len(result) != len(rec)')
    final = [result, rec_aids]
    return session, final


def suggest_carts(df):
    # USE USER HISTORY AIDS AND TYPES
    session = df[0]
    aids = df[1]
    types = df[2]
    
    unique_aids = list(dict.fromkeys(aids[::-1]))
    unique_clicks_carts = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [0, 1]][::-1]))
    unique_carts_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=CANDIDATES:
        weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        
        aids2 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buy2buy]))
        for aid in aids2: aids_temp[aid] += 0.1
        aids3 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buys]))
        for aid in aids3: aids_temp[aid] += 0.1
        aids4 = list(itertools.chain(*[w2v_aids[aid][:10] for aid in unique_carts_buys]))
        for aid in aids4: aids_temp[aid] += 0.1
        
        aids7 = list(itertools.chain(*[bpr_aids[aid][1:4] for aid in unique_aids]))
        for aid in aids7: aids_temp[aid] += 0.1
        
        aids8 = [aid for aid in bpr_sessions[session] if aid not in unique_aids][:3]
        for aid in aids8: aids_temp[aid] += 0.1
        
        sorted_aids = [k for k,v in aids_temp.most_common(CANDIDATES)]
        
        max_count = aids_temp.most_common(1)[0][1]
        min_count = aids_temp.most_common(CANDIDATES)[-1][1]
        # for k in aids_temp: aids_temp[k] /= max_count
        for k in aids_temp: aids_temp[k] = 1 + 0.5*(max_count - min_count)/(max_count-min_count)
        top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES)]
        final = [sorted_aids, top_aids_cnt]
        
        return session, final
            
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids_temp = Counter() 
    aids1 = list(itertools.chain(*[top_20_clicks[aid][:CLICK_LIM] for aid in unique_aids if aid in top_20_clicks]))
    for aid in aids1: aids_temp[aid] += 0.1
    
    aids2 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_aids if aid in top_20_buys]))
    for aid in aids2: aids_temp[aid] += 0.1
    
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_aids if aid in top_20_buy2buy]))
    for aid in aids3: aids_temp[aid] += 0.1
    
    aids4 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buy2buy]))
    for aid in aids4: aids_temp[aid] += 0.1
    
    aids5 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buys]))
    for aid in aids5: aids_temp[aid] += 0.1 
    
    aids6 = list(itertools.chain(*[w2v_aids[aid][:10] for aid in unique_aids]))
    for aid in aids6: aids_temp[aid] += 0.1
    
    aids7 = list(itertools.chain(*[bpr_aids[aid][1:4] for aid in unique_aids]))
    for aid in aids7: aids_temp[aid] += 0.1
    
    aids8 = [aid for aid in bpr_sessions[session] if aid not in unique_aids][:3]
    for aid in aids8: aids_temp[aid] += 0.1
        
    # RERANK CANDIDATES
    top_aids2 = [aid for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    # top_aids2 = [aid for aid, cnt in Counter(aids1 + aids2 + aids3 + aids4 + aids5).most_common(CANDIDATES) if aid not in unique_aids] 
    result = unique_aids + top_aids2[:CAND_FRAC - len(unique_aids)]  
    result = result + [aid for aid in top_carts if aid not in result]
    result = result[:CANDIDATES]
    if len(result) < CANDIDATES:
        print(f'{len(result)} < {CANDIDATES}: session {session}')
        logger.info(f'PROBLEM!!! CARTS: {len(result)} < {CANDIDATES}: session {session}')
    
    max_count = 1.0 * aids_temp.most_common(1)[0][1]
    for k in aids_temp: aids_temp[k] /= max_count
    top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    rec_aids = [1.5 for i in range(len(unique_aids))] + top_aids_cnt
    rec_aids = rec_aids[:CANDIDATES] 
    if len(result) != len(rec_aids):
        rec_aids = (rec_aids + CANDIDATES * [0.0])[:CANDIDATES]
    if len(result) != len(rec_aids):
        print(f'PROBLEM!!! SUGGEST_CARTS - len(result) != len(rec)')
    final = [result, rec_aids]
    return session, final


def suggest_buys(df):
    # USE USER HISTORY AIDS AND TYPES
    session = df[0]
    aids = df[1]
    types = df[2]
    
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    unique_clicks_carts = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [0, 1]][::-1]))
    unique_carts_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=CANDIDATES:
        weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        
        aids2 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buy2buy]))
        for aid in aids2: aids_temp[aid] += 0.1
        aids3 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buys]))
        for aid in aids3: aids_temp[aid] += 0.1
        aids4 = list(itertools.chain(*[w2v_aids[aid][:10] for aid in unique_carts_buys]))
        for aid in aids4: aids_temp[aid] += 0.1
        
        sorted_aids = [k for k,v in aids_temp.most_common(CANDIDATES)]
        
        max_count = aids_temp.most_common(1)[0][1]
        min_count = aids_temp.most_common(CANDIDATES)[-1][1]
        # for k in aids_temp: aids_temp[k] /= max_count
        for k in aids_temp: aids_temp[k] = 1 + 0.5*(max_count - min_count)/(max_count-min_count)
        top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES)]
        final = [sorted_aids, top_aids_cnt]
        
        return session, final
            
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids_temp = Counter() 
    aids1 = list(itertools.chain(*[top_20_clicks[aid][:CLICK_LIM] for aid in unique_aids if aid in top_20_clicks]))
    for aid in aids1: aids_temp[aid] += 1
    
    aids2 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_aids if aid in top_20_buys]))
    for aid in aids2: aids_temp[aid] += 1
    
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_aids if aid in top_20_buy2buy]))
    for aid in aids3: aids_temp[aid] += 1
    
    aids4 = list(itertools.chain(*[top_20_buy2buy[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buy2buy]))
    for aid in aids4: aids_temp[aid] += 1
    
    aids5 = list(itertools.chain(*[top_20_buys[aid][:BUYS_LIM] for aid in unique_carts_buys if aid in top_20_buys]))
    for aid in aids5: aids_temp[aid] += 1
    
    aids6 = list(itertools.chain(*[w2v_aids[aid][:10] for aid in unique_aids]))
    for aid in aids6: aids_temp[aid] += 1
    
    aids7 = list(itertools.chain(*[bpr_aids[aid][1:4] for aid in unique_aids]))
    for aid in aids7: aids_temp[aid] += 1
    
    aids8 = [aid for aid in bpr_sessions[session] if aid not in unique_aids][:3]
    for aid in aids8: aids_temp[aid] += 1
    

    # RERANK CANDIDATES
    top_aids2 = [aid for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    # top_aids2 = [aid for aid, cnt in Counter(aids1 + aids2 + aids3 + aids4 + aids5).most_common(CANDIDATES) if aid not in unique_aids] 
    result = unique_aids + top_aids2[:CAND_FRAC - len(unique_aids)]  
    result = result + [aid for aid in top_carts if aid not in result]
    result = result[:CANDIDATES]
    if len(result) < CANDIDATES:
        print(f'{len(result)} < {CANDIDATES}: session {session}')
        logger.info(f'PROBLEM!!! CARTS: {len(result)} < {CANDIDATES}: session {session}')
        
    max_count = 1.0 * aids_temp.most_common(1)[0][1]
    for k in aids_temp: aids_temp[k] /= max_count
    top_aids_cnt = [cnt for aid, cnt in aids_temp.most_common(CANDIDATES) if aid not in unique_aids]
    rec_aids = [1.5 for i in range(len(unique_aids))] + top_aids_cnt
    rec_aids = rec_aids[:CANDIDATES] 
    if len(result) != len(rec_aids):
        rec_aids = (rec_aids + CANDIDATES * [0.0])[:CANDIDATES]
    if len(result) != len(rec_aids):
        print(f'PROBLEM!!! SUGGEST_BUYS - len(result) != len(rec)')
    final = [result, rec_aids]
    return session, final


def suggest(cfg, data_path, load_path, event):

    # Switch from event "orders" to "buys"
    if event == 'orders':
        event = 'buys'
    
    # Update cutoff lims
    if event == 'clicks':
        click_lim = cfg.baseline_candidates.lims.clicks.click
        buy_lim = cfg.baseline_candidates.lims.clicks.buy
        b2b_lim = cfg.baseline_candidates.top_n.clicks
    elif event == 'carts':
        click_lim = cfg.baseline_candidates.lims.carts.click
        buy_lim = cfg.baseline_candidates.lims.carts.buy
        b2b_lim = cfg.baseline_candidates.top_n.carts
    elif event == 'buys':
        click_lim = cfg.baseline_candidates.lims.buys.click
        buy_lim = cfg.baseline_candidates.lims.buys.buy
        b2b_lim = cfg.baseline_candidates.lims.buys.b2b
        
        
    global CLICK_LIM
    CLICK_LIM = click_lim
    global B2B_LIM
    B2B_LIM = b2b_lim
    global BUYS_LIM
    BUYS_LIM = buy_lim
    del click_lim, b2b_lim, buy_lim
    print(f'CLICK_LIM:{CLICK_LIM}; B2B_LIM{B2B_LIM}; BUYS_LIM:{BUYS_LIM}')
    logger.info(f'CLICK_LIM:{CLICK_LIM}; B2B_LIM{B2B_LIM}; BUYS_LIM:{BUYS_LIM}')
    
    # set global on type_weight_multipliers
    global type_weight_multipliers
    type_weight_multipliers = (dict([(int(key), value) for key, value in 
                                     cfg.type_weight_multipliers.__dict__.items()]))
    
    # Load Co-Visitation Matrices from Disk
    start_time = time.time()
    if cfg.baseline_candidates.debug:
        # TURN ON FOR QUICK DEBUGGING OR ELSE WILL BE INCORRECT
        print('Loading a PARTIAL amount of Candidates for Debugging Purposes')
        top_20_clicks_, top_20_buy2buy_, top_20_buys_ = load_cvm_partial(save_path=load_path,
                                                                tech=cfg.co_vis_matrix)
    else:
        print('Loading a ALL Candidates for Full Runs')
        top_20_clicks_, top_20_buy2buy_, top_20_buys_ = load_cvm(save_path=load_path,
                                                                tech=cfg.co_vis_matrix)
    
    print(f'Load CVM Files: {round((time.time() - start_time) / 60, 3)} mins')
    print('Len. of Three co-visitation matrices:')
    print(f'\tTop 20 Clicks: {len(top_20_clicks_):,}')
    print(f'\tTop 20 Buy2Buy: {len(top_20_buy2buy_):,}')
    print(f'\tTop 20 Buys: {len(top_20_buys_):,}')
    logger.info(f'Load CVM Files: {round((time.time() - start_time) / 60, 3)} mins')
    logger.info('Len. of Three co-visitation matrices:')
    logger.info(f'\tTop 20 Clicks: {len(top_20_clicks_):,}')
    logger.info(f'\tTop 20 Buy2Buy: {len(top_20_buy2buy_):,}')
    logger.info(f'\tTop 20 Buys: {len(top_20_buys_):,}')
        
    # Load Test Data
    test_df = load_test(file_path=data_path.test, type_labels=cfg.type_labels.__dict__)
    
    # Update global
    global CANDIDATES
    CANDIDATES = cfg.candidates
    global CAND_FRAC
    CAND_FRAC = int(cfg.candidates * cfg.cand_percent)
    
    print(f'Number of Candidates: {CANDIDATES}')
    print(f'Num. of Candidates to get from Common Aids: {CAND_FRAC} ({cfg.cand_percent})')
    
    # TOP CLICKS AND ORDERS IN TEST
    top_clicks_ = list(test_df.loc[test_df['type']==0,'aid'].value_counts().index.values[:CANDIDATES])
    top_carts_ = list(test_df.loc[test_df['type']==1,'aid'].value_counts().index.values[:CANDIDATES])
    top_orders_ = list(test_df.loc[test_df['type']==2,'aid'].value_counts().index.values[:CANDIDATES])
    
    # Top Aids from Matrix Factorization
    if cfg.approach == 'validation':
        nn_aids_ = pd.read_parquet('./output/matrix-fac/validation/nn20.parquet').values
        bpr_aids_= pd.read_parquet('./output/bpr/validation/aids_bpr.parquet').values
        bpr_sessions_ = pd.read_parquet('./output/bpr/validation/sess_rec.parquet').values
    else:
        nn_aids_ = pd.read_parquet('./output/matrix-fac/test/nn20.parquet').values
        bpr_aids_= pd.read_parquet('./output/bpr/test/aids_bpr.parquet').values
        bpr_sessions_ = pd.read_parquet('./output/bpr/test/sess_rec.parquet').values
    print(f'nn_aids shape: {nn_aids_.shape}')
    w2v_aids_ = pd.read_parquet('./output/word2vec/test/aid_w2v.parquet').values
    print(f'w2v_aids shape: {w2v_aids_.shape}')
    
    # Update globals
    global top_20_clicks
    top_20_clicks = top_20_clicks_
    global top_20_buy2buy
    top_20_buy2buy = top_20_buy2buy_
    global top_20_buys
    top_20_buys = top_20_buys_
    global top_clicks
    top_clicks = top_clicks_
    global top_orders
    top_orders = top_orders_
    global top_carts
    top_carts = top_carts_
    global nn_aids
    nn_aids = nn_aids_
    global w2v_aids
    w2v_aids = w2v_aids_
    global bpr_aids
    bpr_aids = bpr_aids_
    global bpr_sessions
    bpr_sessions = bpr_sessions_
    
    del top_20_clicks_, top_20_buy2buy_, top_20_buys_, top_clicks_, top_orders_, top_carts_, nn_aids_
    _ = gc.collect()

    PIECES = 5
    valid_bysession_list = []
    for PART in range(PIECES):
        base_path = cfg.otto_valid_test_list.base
        ver = cfg.otto_valid_test_list.val_ver
        if cfg.approach == 'validation':
            group_name = '/valid_group_tolist_'
        else:
            group_name = '/test_group_tolist_'
        with open(f'{base_path}{group_name}{PART}_{ver}.pkl', 'rb') as f:
            valid_bysession_list.extend(pickle.load(f))
    print(f'valid_bysession_list len: {len(valid_bysession_list):,}')
    logger.info(f'valid_bysession_list len: {len(valid_bysession_list):,}')
    
    # Create the Candidates
    start_time = time.time()
    if event == 'clicks':
        temp = df_parallelize_run(suggest_clicks, valid_bysession_list)
        top_n = cfg.co_vis_matrix.click.top_n
    elif event == 'carts':
        temp = df_parallelize_run(suggest_carts, valid_bysession_list)
        top_n = cfg.co_vis_matrix.cart_orders.top_n
    else:
        temp = df_parallelize_run(suggest_buys, valid_bysession_list)
        top_n = cfg.co_vis_matrix.buy_to_buy.top_n
    print(f'suggest {event} loop {round((time.time() - start_time) / 60, 3)} mins')
    logger.info(f'suggest {event} loop {round((time.time() - start_time) / 60, 3)} mins')
        
    # Features
    rec = pd.Series([f[1][0]  for f in temp], index=[f[0] for f in temp]).to_frame().reset_index().rename(columns={0: 'aid', 'index': 'session'})
    rec['rec'] = pd.Series([f[1][1]  for f in temp])
    feat = rec.explode(['aid', 'rec'])
    save_rec = Path('./output/candidates') / f'{cfg.approach}' / \
            f'{event}_topn{top_n}_cand{CANDIDATES}_explode.parquet'
    feat.to_parquet(save_rec)
    print(f'\tSaved Exploded Candidates at: {save_rec}')
    logger.info(f'\tSaved Exploded Candidates at: {save_rec}')
    
    del top_20_clicks, top_20_buy2buy, top_20_buys, top_clicks
    del top_orders, top_carts, nn_aids, rec
    _ = gc.collect()
    
    return feat, save_rec


def suggest_load_existing(bls, approach: str, num_cand: int, event: str):
    # Switch from event "orders" to "buys"
    if event == 'orders':
        event = 'buys'
    # Base path
    base_path = Path(bls.path) / f'{approach}'

    # Top-N
    top_n = bls.top_n.__dict__
    
    # Load Val Clicks
    load_path = base_path / f'{event}_topn{top_n[event]}_cand{num_cand}_explode.parquet'
    df = pd.read_parquet(load_path)
    
    # Reduce datasize for quicker debugging
    if bls.partial:
        df = df[df.session.isin(df.session.unique()[:20])]
    
    return df, load_path
