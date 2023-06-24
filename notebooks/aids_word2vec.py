import gc
import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np
from pathlib import Path
import pandas as pd
from utils.data.load_data import cache_data_to_cpu
from gensim.similarities.annoy import AnnoyIndexer
        

APPROACHES = ['test']
TOPN = 50
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
        
        train_test = pd.concat([train, test])
        train_test = train_test.sort_values(by=['session', 'ts'], ascending=[True, True])
        sentences_df = pl.from_pandas(train_test)
        sentences_df =  sentences_df.groupby('session').agg(pl.col('aid').alias('sentence'))
        sentences = sentences_df['sentence'].to_list()
        del sentences_df, train_test
        _ = gc.collect() 
        
        print('check point')
        
        # Train w2v model
        w2vec = Word2Vec(sentences=sentences,
                 vector_size=64,
                 window=3,
                 negative=8,
                 ns_exponent=0.2,
                 sg=1,
                 min_count=1,
                 workers=8)
        w2vec.save(f'./output/word2vec/{approach}/word2vec.model')
        
        # Indexer
        annoy_index = AnnoyIndexer(w2vec, 300)
        annoy_index.save(f'./output/word2vec/{approach}/indexer/ann')
        
        # # Load
        # w2vec = Word2Vec.load(f'./output/word2vec/{approach}/word2vec.model')
        # w2vec.workers = 8
        # annoy_index = AnnoyIndexer()
        # annoy_index.load(f'./output/word2vec/{approach}/indexer/ann')
        # annoy_index.model = w2vec
        # annoy_index.model.workers = 8

        # w2vec_sim = np.empty([max(unique_aids) + 1, TOPN], dtype=int)
        w2vec_sim = np.empty([len(unique_aids), TOPN], dtype=int)
        cols = [f'aid_{ii}' for ii in range(w2vec_sim.shape[1])]
        LOG_EVERY_N = 50_000
        for i, unique_aid in enumerate(unique_aids):
            sim_aids_scores = w2vec.wv.most_similar([unique_aid],
                                                    topn=TOPN + 1,
                                                    indexer=annoy_index)
            w2vec_sim[unique_aid] = list(zip(*sim_aids_scores))[0][1:]
            if (i % LOG_EVERY_N) == 0:
                print(f'i: {i:,}')
                
        df = pd.DataFrame(w2vec_sim, columns=cols, dtype=int)
        df.to_parquet(f'./output/word2vec/{approach}/aid_w2v.parquet')
        
        print(f'end: {approach}')
        print('check point')