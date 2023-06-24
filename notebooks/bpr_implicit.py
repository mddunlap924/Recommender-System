import gc
import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np
from pathlib import Path
import pandas as pd
from utils.data.load_data import cache_data_to_cpu
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import coo_matrix
        

APPROACHES = ['validation', 'test']
TOPN = 50
ONES = False

if __name__ == "__main__":
    
    for approach in APPROACHES:
        if approach == 'validation':
            base_path = './data/otto-validation'
        elif approach == 'test':
            base_path = './data/otto-chunk-data-inparquet-format'
        
        print(f'start: {approach}')
        # Cache data to RAM

        train_cache, _ = cache_data_to_cpu(data_path=base_path, data_seg='train')
        test_cache, _ = cache_data_to_cpu(data_path=base_path, data_seg='test')

        # Go from dictionary to single dataframe
        train = pd.concat([df for _, df in train_cache.items()])
        test = pd.concat([df for _, df in test_cache.items()]) 

        unique_aids = train.aid.unique().tolist() + test.aid.unique().tolist()
        unique_aids = np.sort(np.unique(unique_aids))
        
        print(f'Train Unique Aids: {train.aid.nunique():,}')
        print(f'Train Max Aids: {train.aid.max():,}')
        print(f'Train Min Aids: {train.aid.min():,}')
        print(f'Test Unique Aids: {test.aid.nunique():,}')
        print(f'Test Max Aids: {test.aid.max():,}')
        print(f'Test Min Aids: {test.aid.min():,}')
        print(f'Unique Aids: {len(unique_aids):,}')
        print(f'Max Aids: {max(unique_aids):,}')
        print(f'Min. Aids: {min(unique_aids):,}')
        
        df = pd.concat([train, test])
        # df = test.copy()
        df = df.groupby(['session', 'aid']).size().reset_index(name='cc')
        df = df.drop_duplicates(subset=['session', 'aid'])
        session_min = df['session'].min()
        aid_min = df['aid'].min()
        df['session'] = df['session'] - session_min
        df['aid'] = df['aid'] - aid_min
        row = df['session'].values
        col = df['aid'].values
        if ONES:
            data = np.ones(df.shape[0])
        else:
            data = df['cc'].values

        ALL_USERS = df['session'].unique().tolist()
        ALL_ITEMS = df['aid'].unique().tolist()
        # coo_train = coo_matrix((data, (row, col)), shape=(df.shape[0], df.shape[0]))
        coo_train = coo_matrix((data, (row, col)), shape=(row.max() + 1, col.max() + 1))
        coo_train = coo_train.tocsr()
        
        model = BayesianPersonalizedRanking(use_gpu=True, random_state=42)
        model.fit(coo_train)
        
        # Get recommendations for the a single user
        userid = 0
        ids, scores = model.recommend(userid,
                                      coo_train[userid],
                                      N=10,
                                      filter_already_liked_items=False)
        # User-item feature
        session_factors = model.user_factors.to_numpy()
        aid_factors = model.item_factors.to_numpy()
        results = df[['session', 'aid']].copy()
        results = results.drop_duplicates()
        
        def user2item(vec1, vec2):
            return np.dot(vec1, vec2)
        
        results['u2i'] = results.apply(lambda x: user2item(vec1=session_factors[x['session']],
                                               vec2=aid_factors[x['aid']]), axis=1)
        # Save feature to disk
        if ONES:
            save_dir = Path(f'./output/bpr/{approach}/ones')
        else:
            save_dir = Path(f'./output/bpr/{approach}')
        results.to_parquet(save_dir / 'user_item.parquet')
        print(f'Saved: {save_dir / "user_item.parquet"}')
        del session_factors, aid_factors, results
        _ = gc.collect()
        
        # Make recommendations for all users in the dataset
        userids = np.arange(df.session.min(), df.session.max() + 1, 1)
        ids, scores = model.recommend(userids, coo_train[userids], N=30)
        cols = [f'session_recs_{i}' for i in range(ids.shape[1])]
        sess_rec = pd.DataFrame(ids, columns=cols)
        sess_rec.to_parquet(save_dir / 'sess_rec.parquet')
        print(f'Saved: {save_dir / "sess_rec.parquet"}')
        del sess_rec, cols, ids, scores
        _ = gc.collect()
        
        # Recommend similar aids for all users in the dataset
        aidids = np.arange(df.aid.min(), df.aid.max() + 1, 1)
        ids, scores = model.similar_items(aidids, N=30)
        cols = [f'aids_bpr_{i}' for i in range(ids.shape[1])]
        aids_bpr = pd.DataFrame(ids, columns=cols)
        aids_bpr.to_parquet(save_dir / 'aids_bpr.parquet')
        print(f'Saved: {save_dir / "aids_bpr.parquet"}')
        del aids_bpr, cols
        _ = gc.collect()
        
        print(f'end: {approach}')
        print('check point')
