import pandas as pd, numpy as np
from tqdm.notebook import tqdm


SAVE = False
NORM_FACTOR = 'T'
type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}


# Added norm to timewt_count
def inter_basic(dfi):
    T = (dfi.ts.max() - dfi.ts.min()) / (24 * 60 * 60)
    Tmin = dfi.ts.min()
    dfi.ts -= Tmin
    
    df = dfi.copy()
    df['inter_by_session'] = df.groupby('session')['session'].transform('count')
    df['num_inter_on_aid'] = df.groupby(['session','aid'])['aid'].transform('count')
    
    # Normalized time weights counts of item interactions
    df = timewt_count(df, 'num_inter_on_aid', norm=T)        
    df_funnel = location_funnel(df=dfi.copy(), event='all')    
    
    # CLICKS
    click = dfi.copy()
    click = click[click.type == 0].sort_values(by='session')
    click['num_click_on_aid'] = click.groupby(['session','aid'])['aid'].transform('count')
    click = timewt_count(click, 'num_click_on_aid', norm=T) 
    click = click[['session', 'aid', 'num_click_on_aid', 'num_click_on_aid_tsw_max',
                   'num_click_on_aid_tsw_mean', 'num_click_on_aid_tsw_std']]
    click_funnel = location_funnel(df=dfi.copy(), event='click')
        
    # CARTS
    cart = dfi.copy()
    cart = cart[cart.type == 1].sort_values(by='session')
    cart['num_cart_on_aid'] = cart.groupby(['session','aid'])['aid'].transform('count')
    cart = timewt_count(cart, 'num_cart_on_aid', norm=T)
    cart = cart[['session', 'aid', 'num_cart_on_aid', 'num_cart_on_aid_tsw_max',
                 'num_cart_on_aid_tsw_mean', 'num_cart_on_aid_tsw_std']]
    cart_funnel = location_funnel(df=dfi.copy(), event='cart')
    
    # ORDERS
    order = dfi.copy()
    order = order[order.type == 2].sort_values(by='session')
    order['num_order_on_aid'] = order.groupby(['session','aid'])['aid'].transform('count')
    order = timewt_count(order, 'num_order_on_aid', norm=T)
    order = order[['session', 'aid', 'num_order_on_aid', 'num_order_on_aid_tsw_max', 
                   'num_order_on_aid_tsw_mean', 'num_order_on_aid_tsw_std']]
    order_funnel = location_funnel(df=dfi.copy(), event='order')

    # merge funnel info.
    df = df.merge(df_funnel, on=['session', 'aid'], how='left')
    click = click.merge(click_funnel, on=['session', 'aid'], how='left')
    cart = cart.merge(cart_funnel, on=['session', 'aid'], how='left')
    order = order.merge(order_funnel, on=['session', 'aid'], how='left')
    
    for df_ in [click, cart, order]:
        df = df.merge(df_, on=['session', 'aid'], how='left').fillna(0)
    
    df['inter_percent_inter_aid'] = df['num_inter_on_aid'] / T
    df['inter_percent_click_aid'] = df['num_click_on_aid'] / T
    df['inter_percent_cart_aid'] = df['num_cart_on_aid'] / T
    df['inter_percent_order_aid'] = df['num_order_on_aid'] / T
    df.drop(columns=['num_inter_on_aid',
                     'num_click_on_aid',
                     'num_cart_on_aid',
                     'num_order_on_aid',
                     'ts_norm','cc_weights', 'inter_by_session'],
            inplace=True)
    df = df.drop_duplicates(subset=['session', 'aid'])
    df.drop(columns=['type', 'ts'], inplace=True)
    
    return df


def inter_hour_day(df):
    df['tsd'] = pd.to_datetime(df['ts'], unit='s')
    df['day'] = df.tsd.dt.dayofweek
    df['hour'] = df.tsd.dt.hour
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    all_type_labels = {'all': None}
    all_type_labels.update(type_labels)
    for event, label in all_type_labels.items():
        if event == 'all':
            tmp = df.copy()
        else:
            tmp = df[df.type == label].copy()
        dfc = (tmp.groupby(['session', 'aid'])
            .agg({'day': ['nunique', 'mean', 'min', 'max'],
                    'hour': ['nunique', 'mean', 'min', 'max'],
                    }))
        dfc.columns = [f'inter_{event}_' + '_'.join(col).strip() for col in dfc.columns.values]
        for col in dfc.columns.tolist():
            if 'nunique' in col:
                dfc[col] = dfc[col] / T
        df = df.merge(dfc, on=['session', 'aid'], how='left').fillna(0)
        df.drop_duplicates(subset=['session', 'aid'], inplace=True)
    df.drop(columns=['type', 'ts', 'tsd', 'day', 'hour'], inplace=True)
    return df


def ses_aid_interaction_time(dft, event, save_base, approach):
    
    df = dft[['session', 'aid', 'type']]       
    df = df[["session", "aid"]].loc[df.type == type_labels[event]].drop_duplicates(subset=["session","aid"]).reset_index(drop=True)
    df[f"inter_item_{event}"] = 1
    
    df_time = dft.merge(df.copy(), on=['session','aid'], how='left')
    
    T = (df_time.ts.max() - df_time.ts.min()) / (24 * 60 * 60)
    Tmin = df_time.ts.min()
    df_time.ts -= Tmin
    
    df_time = df_time[df_time.type == type_labels[event]].reset_index(drop=True)
    
    df_dup = df_time[['session', 'aid', 'ts']][df_time.duplicated(subset=['session', 'aid'], keep=False)].copy()
    df_dup[f'{event}_first_ts'] = df_dup.groupby(['session', 'aid'])['ts'].transform('min')
    df_dup[f'{event}_last_ts'] = df_dup.groupby(['session', 'aid'])['ts'].transform('max') - df_dup[f'{event}_first_ts']
    df_dup[f'{event}_avg_ts'] = df_dup.groupby(['session', 'aid'])['ts'].transform('mean')
    df_dup.drop_duplicates(subset=['session', 'aid'], inplace=True)
    df_dup.drop(columns=['ts'], inplace=True)
    print(f'{df_dup.shape[0]:,}')

    df_notdup = df_time.drop_duplicates(subset=['session', 'aid'], keep=False).copy()
    df_notdup[f'{event}_first_ts'] = df_notdup['ts'].copy()
    df_notdup[f'{event}_last_ts'] = df_notdup['ts'].copy() - df_notdup[f'{event}_first_ts']
    df_notdup[f'{event}_avg_ts'] = df_notdup['ts'].copy()
    
    df_time = pd.concat([df_dup, df_notdup]).sort_values(by='session')
    df_time = df_time[['session', 'aid', f'{event}_first_ts', f'{event}_last_ts', f'{event}_avg_ts']]
    
    print(f'shape {event}: {df.shape}')
    df = df.merge(df_time, on=['session', 'aid'])
    print(f'\tafter: {df.shape}')
    #TODO check output here something is wrong look at first and last ts (last can be before first)
    df.to_parquet(save_base / approach / 'interaction' / f'item_{event}.parquet')
    del df, df_dup, df_time, df_notdup
    
    return


def session_ts_bw(dft):
    df = dft.copy()[['session']].sort_values(by=['session'])
    df = df.drop_duplicates()
    events = [[['click', 0], ['cart', 1]],
                [['click', 0], ['order', 2]],
                [['cart', 1], ['order', 2]]
                ]
    for event in events:
        tmp = dft.copy()
        T = (tmp.ts.max() - tmp.ts.min()) / (24 * 60 * 60)
        Tmin = tmp.ts.min()
        tmp.ts -= Tmin
        aname, aval = event[0][0], event[0][1]
        bname, bval = event[1][0], event[1][1]
        tmp = tmp[tmp['type'].isin([aval, bval])].sort_values(by=['session', 'ts', 'type'])
        a = tmp.drop_duplicates(subset=['session', 'type']).sort_values(by=['session', 'type'])
        a = a[a['type'] == aval][['session', 'ts', 'type']]
        b = tmp.drop_duplicates(subset=['session', 'type']).sort_values(by=['session', 'type'])
        b = b[b['type'] == bval][['session', 'ts', 'type']]
        c = a.merge(b, on='session', how='outer')
        c[f'session_{aname}_{bname}_ts_diff'] = c['ts_y'] - c['ts_x']
        c = c[['session', f'session_{aname}_{bname}_ts_diff']]
        df = df.merge(c, on='session', how='left')
    df = df.fillna(0.0)
    return df


def location_funnel(df, event):
    # Product funnel where aids first and last occurs in a users interactions
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    if event == 'click':
        df = df[df.type == 0].sort_values(by=['session', 'ts'], ascending=[True, True])
    elif event == 'cart':
        df = df[df.type == 1].sort_values(by=['session', 'ts'], ascending=[True, True])
    elif event == 'order':
        df = df[df.type == 2].sort_values(by=['session', 'ts'], ascending=[True, True])
    else:
        df = df.sort_values(by=['session', 'ts'], ascending=[True, True])
    df['inter_by_session'] = df.groupby('session')['session'].transform('count')
    df['cc'] = df.groupby('session').transform('cumcount') + 1
    df[f'inter_aid_{event}_first_occ'] = df.groupby(['session', 'aid'])['cc'].transform('first')
    df[f'inter_aid_{event}_last_occ'] = df.groupby(['session', 'aid'])['cc'].transform('last')
    df[f'inter_aid_{event}_avg_occ'] = df.groupby(['session', 'aid'])['cc'].transform('mean')
    df[f'inter_aid_{event}_first_occ_ts'] = df.groupby(['session', 'aid'])['ts'].transform('first')
    df[f'inter_aid_{event}_last_occ_ts'] = df.groupby(['session', 'aid'])['ts'].transform('last')
    df[f'inter_aid_{event}_avg_occ_ts'] = df.groupby(['session', 'aid'])['ts'].transform('mean')
    
    # Normalize
    norm_fact = df['inter_by_session']
    df[f'inter_percent_aid_{event}_first_occ'] = df[f'inter_aid_{event}_first_occ'] / norm_fact
    df[f'inter_percent_aid_{event}_last_occ'] = df[f'inter_aid_{event}_last_occ'] / norm_fact
    df[f'inter_percent_aid_{event}_avg_occ'] = df[f'inter_aid_{event}_avg_occ'] / norm_fact
    
    df.drop(columns=['cc', 'ts', 'inter_by_session', 'type',
                     f'inter_aid_{event}_first_occ',
                     f'inter_aid_{event}_last_occ',
                     f'inter_aid_{event}_avg_occ'], inplace=True)
    df = df.drop_duplicates(subset=['session', 'aid'])
    return df


def num_clicks_before(df, event1, event2):
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    df['inter_by_session'] = df.groupby('session')['session'].transform('count')
    df['num_inter_on_aid'] = df.groupby(['session','aid'])['aid'].transform('count')
        
    df = df.sort_values(by=['session', 'ts'], ascending=[True, True])
    a, b = df[df['type'] == event1], df[df['type'] == event2]
    c = a.merge(b, on=['session', 'aid'], how='left').dropna(subset=['type_y'])
    c = c[['session', 'aid']].value_counts().reset_index(level=[0,1])
    c.rename(columns={0: f'num_{event1}_before_{event2}'}, inplace=True)
    return c


def timewt_count(df, col, norm):
    df['ts_norm'] = df['ts'] / df.groupby(['session'])['ts'].transform('max')
    df['ts_norm'] = df['ts_norm'].replace(np.nan, 0.0)
    df['cc_weights'] = df[col] + df['ts_norm']
    df[f'{col}_tsw_max'] = df.groupby(['session', 'aid'])['cc_weights'].transform('max') / norm
    df[f'{col}_tsw_mean'] = df.groupby(['session', 'aid'])['cc_weights'].transform('mean') / norm
    df[f'{col}_tsw_std'] = df.groupby(['session', 'aid'])['cc_weights'].transform('std') / norm
    df[f'{col}_tsw_std']= df[f'{col}_tsw_std'].fillna(0.0)
    return df
