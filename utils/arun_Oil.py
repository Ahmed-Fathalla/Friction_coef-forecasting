import pandas_datareader as pdr
import sys
import pandas as pd
import numpy as np

# from keras.models import save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils._utils import *
from utils.time_utils import *
from utils.HP_optim import *
from utils.pkl_utils import *
from utils.scoring import *
from utils.conf import *
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.Callbacks.test_data_callbacks import test_data_callbacks
import os
from utils.metrics import POCID, u_theil
from utils.metrics import get_res
from utils.Callbacks.callback_utils import df_to_str
# hyperopt_space = \
#     {
#         'nb_epochs' : hp.choice('nb_epochs',[70]), #hp.choice('nb_epochs',[50]),
#         'GRU_cells' : hp.choice('GRU_cells',[32]),
#         'd1_nodes' : hp.choice('d1_nodes',[16]),
#         'd2_nodes' : hp.choice('d2_nodes',[4]),
#         'batch_size' : hp.choice('batch_size',[16,32]),
#         'dropout1': hp.choice('dropout1',[0,0.1,0.2]),
#         'dropout2': hp.choice('dropout2',[0,0.1,0.2]),
#         'activation_1': hp.choice('activation_1',['relu']),
#         'activation_2': hp.choice('activation_2',['relu']),
#         'hyperopt_max_trials': 20,
#     }
metrics__ =['mae','mse', 'rmse', 'R2' ,POCID, u_theil]# , MAPE_Other
metrics =['mae','mse', POCID, u_theil]
def save_pred(y_true, y_pred, scaler, file_name, Apply_diff=0, orig=None):
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

    true_vs_pred_values = pd.DataFrame(
                                        np.hstack([np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)]),
                                        columns='y_test,y_pred'.split(',')
                                      )

    print('get_res(metrics, a, b) = ', get_res(y_true, y_pred, metric_lst=metrics__))
    if Apply_diff:
        true_vs_pred_values['error'] = true_vs_pred_values['y_test'] - true_vs_pred_values['y_pred']
        true_vs_pred_values['orig'] = orig
        true_vs_pred_values['pred_inv_diff'] = true_vs_pred_values['orig'] - true_vs_pred_values['error']
        # true_vs_pred_values['error2'] = true_vs_pred_values['orig'] - true_vs_pred_values['pred_inv_diff']

        print('^^^^ inv_diff '+' '*11, get_res(true_vs_pred_values['orig'], true_vs_pred_values['pred_inv_diff'], metric_lst=metrics__))
        print()
        true_vs_pred_values.to_csv(file_name)
        return get_res(true_vs_pred_values['orig'], true_vs_pred_values['pred_inv_diff'], metric_lst=metrics__)

    true_vs_pred_values.to_csv(file_name)
    return get_res(y_true, y_pred, metric_lst=metrics__)

def Run_Oil(scaling=1, blocks=5,trails=5, exp_name='', fast_run=False, window_size=10, Apply_diff =0, hyperopt_space={}):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    df = pd.read_csv('oil_data.csv')

    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        blocks = 1
        trails = 1
        df = df[['Value']][:200]
        epoches = 2
    else:
        exp_str = exp_name + get_TimeStamp_str() + ' '
        df = df[['Value']]
        epoches = 50

    test_ratio = 0.2
    valid_ratio = 0.2

    df = df.rename(columns ={'Value':'y'})

    scaler_ = MinMaxScaler()
    df['y'] = scaler_.fit_transform(df['y'].values.reshape(-1, 1))
    print('Data Range', df['y'].min(), df['y'].max())


    x_data, y_data, scaler = get_data_look_back(df['y'], scaling=scaling, window_size=window_size, verbose=1, Apply_diff=Apply_diff)
    print('get_data_look_back scaler = ', scaler)
    print('after minmax scalling', y_data.min(), y_data.max(), '   mean:', y_data.mean(), '   std:', y_data.std())

    train_len = int(len(x_data)*(1-test_ratio))
    # train_len = records_per_second * minutes_
    x, x_test = x_data[:train_len], x_data[train_len:] # train_len + int(len(x_data)*0.1)
    y, y_test = y_data[:train_len], y_data[train_len:]

    X_train, y_train, X_valid, y_valid = train_test_split(x, y, train_ratio=(1 - valid_ratio))

    y_train_orig, y_val_orig, y_test_orig = None, None, None
    if Apply_diff:
        data = df['y'][1:-1*window_size].values
        print('data.len ----------', len(data))
        y_train_orig = data[: len(y_train) ]
        y_val_orig   = data[len(y_train) : len(y_train) + len(y_valid)]
        y_test_orig  = data[- len(y_test):]

    print('y_train_orig.len ----------', len(y_train_orig))
    print('y_val_orig.len ----------', len(y_val_orig))
    print('y_test_orig.len ----------', len(y_test_orig))

    print('X_train.shape ----------', X_train.shape)
    print('X_valid.shape ----------', X_valid.shape)
    print('x_test.shape ----------', x_test.shape)

    # import sys
    # sys.exit()

    for blocks_count in range(blocks):
        txt_str_ = ''
        txt_str_ += 'Train&val:' + str(train_len) + ' Test:' + str(len(x_test)) + '\n'

        txt_str_ += '  X_train:%r X_valid;%r'%(X_train.shape, X_valid.shape) + '\n'
        setting_dict, str_ = get_best_architecture(X_train, y_train, X_valid, y_valid,
                                                   trails=trails, setup_dict=hyperopt_space, epoches=epoches)
        train_score_lst, val_score_lst, test_score_lst, epoch_lst = [], [], [], []
        fitting_time, prediction_time = '  ', ''
        exp_id = str.strip(exp_str) + ' block_%d'%blocks_count
        os.mkdir(exp_id)
        # exp_id = exp_id[:-1] + str(i)
        # print('##########################3')
        model = model_from_dict(setting_dict, X_train.shape)
        # mod_str = exp_name + get_TimeStamp_str() + ' min_%d trail_%d i_%d'%(minutes_, blocks_count, i) + '.h5'
        patience = 7
        es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
        checkpoint = ModelCheckpoint('%s/model.h5'%exp_id, monitor='val_loss', verbose=True, save_best_only=True, mode='min')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%metrics = ', metrics)
        ts = test_data_callbacks(exp_id = exp_id,
                                 test_data=(x_test, y_test),
                                 metrics_=metrics,
                                 metric_index=0,
                                 round_=7)
        st_time = time()
        hist = model.fit(X_train, y_train,
                         validation_data=(X_valid, y_valid),
                         epochs=setting_dict['nb_epochs'],
                         batch_size=setting_dict['batch_size'],
                         callbacks=[es, checkpoint, ts],
                         shuffle=False,
                         verbose=0
                         )
        fitting_time += '%-.4f'%((time() - st_time) / 60.0) + ', '

        # st_time = time()
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(x_test)
        # prediction_time += '%-.4f'%((time() - st_time) / 60.0) + ', '

        train_score = save_pred(y_train, y_train_pred, scaler, Apply_diff=Apply_diff, file_name='%s/train_pred.csv'%(exp_id), orig=y_train_orig)
        val_score   = save_pred(y_valid, y_valid_pred, scaler, Apply_diff=Apply_diff, file_name='%s/valid_pred.csv'%(exp_id), orig=y_val_orig)
        test_score  = save_pred(y_test,  y_test_pred,  scaler, Apply_diff=Apply_diff, file_name='%s/test_pred.csv'%(exp_id), orig=y_test_orig)

        res = np.array([train_score, val_score, test_score])
        with open('%s/a report.txt'%exp_id, 'a') as myfile:myfile.write( '\n'*4 )
        df_to_str(res,
                  file='%s/a report.txt'%exp_id,
                  print_=True,
                  headers= [i.__name__ if callable(i) else i for i in metrics__])

        return X_train, y_train, X_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred