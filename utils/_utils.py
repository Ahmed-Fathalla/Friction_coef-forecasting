import sys
import numpy as np
import pandas as pd
import warnings;warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import operator
from tabulate import tabulate

def get_data_look_back(dataseries, window_size = 3, reshape_ = 1, Apply_diff = 0, verbose = 0, scaling = 1):
    '''
    this function does the following:
        1. takes a dataseries (vector),
        2. makes its differences, and
        3. create number of 'window_size' lag features
    '''
    
    if not isinstance(dataseries, pd.core.series.Series):
        dataseries = pd.Series(dataseries)

    df = dataseries.to_frame()
    df.columns = ['x_0']

    df.index = range(len(df))
    if Apply_diff:
        df['x_0'] = df['x_0'].diff()#.shift(-1) fat_1
        df = df[1:] # remove the first observation which has a difference of NaN
    else:
        df['x_0'] = df['x_0']

    scaler = None
    if scaling>0:
        if scaling == 1:
            scaler = StandardScaler()
        elif scaling==2:
            scaler = MinMaxScaler()
        df['x_0'] = scaler.fit_transform(df['x_0'].values.reshape(-1,1))
    else:
        ...

    for i in range(1, window_size):
        df['x_'+str(i)] = df.x_0.shift(-1*i)

    col_names = ['x_'+str(i)for i in range(0,window_size)]

    df['y'] = df.x_0.shift(-1*(window_size))
    # df['curr_temp'] = df['x_0'].shift(-1*(window_size)) # (window_size+1) fat_2

    df = df[:-1*(window_size)] # reomve observations of NaN values
    df = df[[*col_names, 'y']] # , 'curr_temp'

    # print(df)

    y_data = df.pop('y')

    if reshape_:
        x_data = np.reshape(df[col_names].values, (df[col_names].shape[0], df[col_names].shape[1], 1))
    else:
        x_data = df[col_names].values

    if verbose:
        print('x_data:', x_data.shape)

    return x_data, y_data, scaler

def sort_tuble(tub, item = 2, ascending = True):
    tub = sorted(tub, key=operator.itemgetter(item), reverse=False)
    if ascending:
        return tub[0]
    else:
        return tub[-1]

def train_test_split(x_data_, y_data_, train_ratio = 0.4):
    train_obs = int(x_data_.shape[0]*train_ratio)
    X_train, y_train = x_data_[:train_obs], y_data_[:train_obs]
    X_valid, y_valid = x_data_[train_obs:], y_data_[train_obs:]
    return X_train, y_train, X_valid, y_valid

def df_to_file(df, round_=5, file=None, padding='left', rep_newlines='\t', print_=False, wide_col='', pre='', post=''):
    headers = [wide_col+str(i)+wide_col for i in df.columns.values]
    c = rep_newlines + tabulate(df.round(round_).values,
                                headers=headers,
                                stralign=padding,
                                disable_numparse=1,
                                tablefmt = 'grid' # 'fancy_grid' ,
                                ).replace('\n', '\n'+rep_newlines)
    if print_:print(c)
    if file is not None:
        with open(file, 'a', encoding="utf-8") as myfile:
            myfile.write( pre + c + post + '\n')

def write(metric_lst, file, txt='', data=None, col=None, models='', get_result=False, ignore_first_raw=False):
    with open(file, 'a') as myfile:
        myfile.write( txt + '\n')

    if data is not None:
        with open(file, 'a') as myfile:
            myfile.write('\n')
        test_score_lst = np.array(data, dtype=float)
        if col is None:
            m_lst = [i.__name__ if callable(i) else i for i in metric_lst]
        else:
            m_lst = col

        test_score_lst = pd.DataFrame(np.array(test_score_lst).reshape(-1,len(m_lst)), columns=m_lst)
        test_score_lst['models'] = models
        df_to_file(test_score_lst, file=file)

        a = test_score_lst[m_lst[:]] # -1

        if ignore_first_raw:
            a = a[1:]

        for i in a.columns[:-1]:
            a[i] = a[i].astype(float)
        a = a.describe()
        a = a.loc[['mean', 'std'], :]
        df_to_file(a, pre='\nStatistical prob:\n', post='\n\n', file=file)
        with open(file, 'a') as myfile:
            myfile.write( '\n\n' )

        if get_result:
            return a.values[0].tolist()

import matplotlib.pyplot as plt
def plt_loss(file, hist, metric):
    from .pkl_utils import dump
    import traceback
    try:
        dump('%s.pkl' % file, hist)
    except Exception as exc:
        pass
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], )
    plt.savefig('%s.pdf'%file, bbox_inches='tight')
    plt.show()