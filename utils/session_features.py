import pandas as pd, numpy as np
import gc


SAVE = False
NORM_FACTOR = 'T'
type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def session_basic(df):
    df['tsd'] = pd.to_datetime(df['ts'], unit='s')
    df['day'] = df.tsd.dt.dayofweek
    df['hour'] = df.tsd.dt.hour
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    dfc = df.groupby('session').agg({'aid': ['count', 'nunique'],
                'type': ['mean', 'min', 'max'],
                'ts': ['min', 'max', 'mean', 'std'],
                'day': ['mean', 'max'],
                'hour': ['mean', 'max'],
                })
    dfc.columns = ['session_' + '_'.join(col).strip() for col in dfc.columns.values]
    dfc.fillna(0.0, inplace=True)
    dfc['session_length_ts'] = dfc['session_ts_max'] - dfc['session_ts_min']
    print(f'\tsession completed first agg')
    
    # Interaction by day-of-week
    df0 = df['session'].copy()
    df0 = df0.to_frame().drop_duplicates(subset='session')
    day_col_names = []
    for day in range(7):
        day_col_name = f'session_inter_day_{day}'
        day_col_names.append(day_col_name)
        tmp = df[df['day'] == day][['session', 'aid']].copy().reset_index(drop=True)
        tmp[day_col_name] = tmp.groupby('session')['aid'].transform('count')
        tmp = tmp[['session', day_col_name]].drop_duplicates(subset='session')
        df0 = df0.merge(tmp, on='session', how='left').fillna(0)
        del tmp
        _ = gc.collect()
    df0 = df0[['session'] + day_col_names]
    df0 = df0.drop_duplicates(subset='session')
    print(f'\tmerge session day')
    dfc = dfc.merge(df0, on='session', how='left')
    print(f'\t\tcompleted merge session day')
    del df0
    _ = gc.collect()
    
    # Percent count of clicks, carts, and orders by aid
    df0 = df.copy()
    df0['inter_count'] = df0.groupby(['session'])['aid'].transform('count')
    cols = []
    for event, label in type_labels.items():
        col = f'session_percent_count_{event}'
        cols.append(col)
        tmp = df0[df0.type == label].groupby(['session'])['aid'].count()
        tmp.name = col             
        df0 = df0.merge(tmp, on='session', how='left')
        del tmp
        _ = gc.collect()
    df0.fillna(0.0, inplace=True)
            
    for col in cols:
        df0[col] = df0[col] / df0['inter_count']
    df0.fillna(0.0, inplace=True)
    df0.drop_duplicates(subset='session', inplace=True)
    df0 = df0[['session'] + cols]
    
    dfc = dfc.merge(df0, on='session', how='left')
    del cols, df0
    _ = gc.collect()
    
    # Number of aids sessions for clicks, carted, and ordered an aid
    cols = []
    for event, label in type_labels.items():
        tmp = df[df.type == label].groupby('session').agg({'aid': ['count', 'nunique']})
        tmp.columns = [f'session_' + '_'.join(col).strip() + f'_{event}' for col in tmp.columns.values]
        for col in tmp.columns.tolist():
            cols.append(col)
        dfc = dfc.merge(tmp, on='session', how='left').fillna(0.0)
        del tmp
        
    # Normalize counts by days
    cols += ['session_aid_count', 'session_aid_nunique', 'session_length_ts']
    cols += day_col_names
    for col in cols:
        # if 'nunique' not in col:
        dfc[col] = dfc[col] / T
    return dfc


def session_ts_diff(df):
    grp_by = ['session', 'aid']
    df['session_ts_diff'] = df.groupby(grp_by)['ts'].transform('diff')
    df['session_ts_diff_avg'] = df['session_ts_diff'] / df.groupby(grp_by)['ts'].transform('mean')
    df['session_ts_diff_std'] = df['session_ts_diff'] / df.groupby(grp_by)['ts'].transform('std')
    df['session_ts_diff_min'] = df['session_ts_diff'] / df.groupby(grp_by)['ts'].transform('min')
    df['session_ts_diff_max'] = df['session_ts_diff'] / df.groupby(grp_by)['ts'].transform('max')
    df.fillna(0, inplace=True)
    df = df[['session', 'session_ts_diff_avg', 'session_ts_diff_std',
             'session_ts_diff_min', 'session_ts_diff_max']]
    df = df.drop_duplicates(subset='session').copy()
    return df


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

    
    # Normalize
    if NORM_FACTOR == 'T':
        norm_fact = T
    else:
        norm_fact = df['inter_by_session']
    df[f'inter_percent_aid_{event}_first_occ'] = df[f'inter_aid_{event}_first_occ'] / norm_fact
    df[f'inter_percent_aid_{event}_last_occ'] = df[f'inter_aid_{event}_last_occ'] / norm_fact
    
    df.drop(columns=['cc', 'inter_by_session', 'type', f'inter_aid_{event}_first_occ', f'inter_aid_{event}_last_occ'], inplace=True)
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


def timewt_count(df, col):
    df['ts_norm'] = df['ts'] / df.groupby(['session'])['ts'].transform('max')
    df['ts_norm'] = df['ts_norm'].replace(np.nan, 0.0)
    df['cc_weights'] = df[col] + df['ts_norm']
    df[f'{col}_ts_sum'] = df.groupby(['session', 'aid'])['cc_weights'].transform('max')
    return df


def session_more(df):
    T = (df.ts.max() - df.ts.min()) / (24 * 60 * 60)
    Tmin = df.ts.min()
    df.ts -= Tmin
    
    # Time of day info
    dts = pd.to_datetime(df['ts'], unit='s') ## pandas recognizes your format

    df['day'] = (dts.dt.weekday).astype('int8')
    df['hour'] = (dts.dt.hour).astype('int8')
    df['hm'] = (dts.dt.hour*100 + dts.dt.minute*100//60).astype('int16')
    df['hm_mean'] = df.groupby('session')['hm'].transform('mean')
    df['hm_std'] = df.groupby('session')['hm'].transform('std')
    
    for i in range(7):
        tmp = df[df['day'] == i].groupby('session')['day'].count().to_frame()
        tmp.rename(columns={'day': f'day{i}cnt'}, inplace=True)
        df = df.merge(tmp, on='session', how='left').fillna(0.0)
        del tmp
        df[f'day{i}cnt'] = df[f'day{i}cnt'] / T
        
    for i in range(24):
        tmp = df[df['hour'] == i].groupby('session')['hour'].count().to_frame()
        tmp.rename(columns={'hour': f'hour{i}cnt'}, inplace=True)
        df = df.merge(tmp, on='session', how='left').fillna(0.0)
        del tmp
        df[f'hour{i}cnt'] = df[f'hour{i}cnt'] / T
    
    return df
