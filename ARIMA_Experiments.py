from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

import gc
import tensorflow as tf
import sys, os
from utils.pkl_utils import dump

from utils._utils import *
from utils.time_utils import *
from utils.HP_optim import *
from utils.scoring import *
from utils.metrics import *
from time import time

metric_lst = [MAE_sklearn, MSE_sklearn, RMSE_sklearn, POCID]

def ARIMA_exp2(minutes_start=[7, 10, 13], exp_name='',
         fast_run=False, dump_model=False, save_csv_pred=False, ds=1):

    df = pd.read_csv('dataset/%s.csv' % str(ds), encoding='latin1')


    if type(minutes_start) is int:
        minutes_start = [minutes_start]
    elif type(minutes_start) is list:
        pass

    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT_Fast/Exp_2 %s_csv ' % ds + exp_str)
        df = df['y'].values[:200]
        records_per_minute = 10
    else:
        exp_str = '' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT/Exp_2 %s_csv ' % ds + exp_str)
        df = df['y'].values
        records_per_minute = 6000
    os.makedirs(folder)

    for minutes in minutes_start:

        split_count = int(minutes * records_per_minute)
        train = df[:split_count]
        test = df[split_count:]

        a = time()
        a_arima = auto_arima(train, start_p=0, start_q=0,
                             stepwise=True, suppress_warnings=True,
                             seasonal=True, stationary=True, trend='c',
                             error_action='ignore', random_state=123)
        auto_t = time() - a

        a = time()
        a_arima.fit(train)
        fit_t = time() - a

        a = time()
        preds = a_arima.predict(n_periods=len(test))
        pred_t = time() - a

        res = get_res(test, preds, metric_lst=metric_lst, round_=5)

        mod_str = 'min_%d.pkl' % (minutes)
        if dump_model:
            dump(folder + '/%s_ARIMA_model.pkl'%(mod_str[:-4]), a_arima)

        if save_csv_pred:
            true_vs_pred_values = pd.DataFrame(
                np.hstack([np.array(test).reshape(-1, 1), np.array(preds).reshape(-1, 1)]),
                columns='y_test,y_pred'.split(','))
            true_vs_pred_values.to_csv(folder + '/%s.csv'%(mod_str[:-4]))

        exp_ = '00.txt'
        file_str = folder + '/' + exp_
        write(metric_lst, file=file_str, txt='min_%d\nGrid:%f\nFitting:%f\nPred:%f\n\n%r\n%s\n'%
                                             (minutes,auto_t,fit_t,pred_t,res,'-'*80))

def ARIMA_exp3(training_mins=None, alternating_window=None, exp_name='',
         fast_run=False, dump_model=False, save_csv_pred=False, ds=1):

    if training_mins is None and alternating_window is None:
        comp = [[0.5, 0.25], [0.5, 1.0], [1.0, 0.5], [1.0, 2.0]]
    else:
        comp = [[training_mins, alternating_window]]


    df = pd.read_csv('ds/%s.csv' % str(ds), encoding='latin1')
    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT_Fast/Exp_3 %s_csv ' % ds + exp_str)
        df = df[['time', 'y']][:50]
        records_per_minute = 10
    else:
        exp_str = exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT/Exp_3 %s_csv ' % ds + exp_str)
        df = df[['time', 'y']]
        records_per_minute = 6000
    os.makedirs(folder)

    for training_mins, alternating_window in comp:
        print('=========> training_mins, alternating_window = ', training_mins, alternating_window)
        file_str = folder + '/%f %f.txt' % (training_mins, alternating_window)

        train_len = int(records_per_minute * training_mins)
        initial_train = df['y'][:train_len]
        y_test = df['y'][train_len:]
        print('1_ y_test.shape ----------', y_test.shape)
        cycle_records = int(alternating_window * records_per_minute)
        cycles = 0

        a = time()
        print('initial_train.len ----------', len(initial_train))
        a_arima = auto_arima(initial_train, start_p=0, start_q=0,
                             stepwise=True, suppress_warnings=True,
                             seasonal=True, stationary=True, trend='c',
                             error_action='ignore', random_state=123)
        a_arima.fit(initial_train)
        b = time() - a

        if dump_model:
            dump('%s %d.pkl' % (file_str[:-4], cycles), a_arima)


        # file = folder + '/' + '%f %f.txt'%(training_mins,alternating_window)
        write(metric_lst, file=file_str, txt = '\ncycles:i:%d  Used model parameteres :%s  time:%f' % (cycles, a_arima, b))
        y_true, y_pred, test_score_lst = [], [], []
        while True:
            period_data = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)]
            print('Cyc_', cycles, 'period_data.shape ----------', period_data.shape)
            y_true.append(period_data)
            y_pred.append(a_arima.predict(n_periods=len(period_data)))

            cycles += 1

            # print('--------------', (cycles + 2) * cycle_records, len(y_test))
            # print('@@@@ if (cycles + 2) * cycle_records > len(y_test) = ', (cycles + 2) * cycle_records, len(y_test), '   ',
            #       cycles)
            if (cycles + 2) * cycle_records > len(y_test):
                break

            updating_train = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)]
            a = time()
            a_arima = auto_arima(updating_train, start_p=0, start_q=0,
                                 stepwise=True, suppress_warnings=True,
                                 seasonal=True, stationary=True, trend='c',
                                 error_action='ignore', random_state=123)
            a_arima.fit(updating_train)

            if dump_model:
                dump('%s %d.pkl' % (file_str[:-4], cycles), a_arima)

            b = time() - a
            # dump('%s %d.pkl'%(file_str[:-4], cycles ), a_arima)
            write(metric_lst, file=file_str,txt='\ncycles:i:%d  Used model parameteres :%s  time:%f' % (cycles, a_arima, b))

            cycles += 1

        for y_true_lst, y_pred_lst in zip(y_true, y_pred):
            y_test_inv = y_true_lst
            y_pred_inv = y_pred_lst
            test_score_lst.append([*get_res(y_test_inv, y_pred_inv, metric_lst=metric_lst)])

        # write(metric_lst, file=file_str, txt=txt_str_ + '\n\n' + str_ + '\n\n')
        print('test_score_lst.len ----------', len(test_score_lst))


        str_ = write(metric_lst, file=file_str,
                     data=test_score_lst,
                     txt='\ncycles:i:%d  Used model parameteres :%s  time:%f' % (cycles, a_arima, b),
                     get_result=True)
        print('str_ = ', str_)
        with open(folder + '/00.txt', 'a') as myfile:
            myfile.write('%r  %f %f' % (str_,training_mins, alternating_window) + '\n')

def ARIMA_exp4(training_mins=0.25, alternating_window=0.5, exp_name='', auto_ar=True,
         fast_run=False, dump_model=False, save_csv_pred=False, ds=1):

    df = pd.read_csv('ds/%s.csv' % str(ds), encoding='latin1')
    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT_Fast/Exp_4 %s_csv ' % ds + exp_str)
        df = df[['time', 'y']][:50]
        records_per_minute = 10
    else:
        exp_str = exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('ARIMA OUTPUT/Exp_4 %s_csv ' % ds + exp_str)
        df = df[['time', 'y']]  # 30000+(int(6000*1.5))
        records_per_minute = 6000
    os.makedirs(folder)
    file_str = folder + '/%f.txt'%training_mins
    cycle_records = alternating_window * records_per_minute

    res_lst = []

    initial_train = df['y'][0: int(training_mins * records_per_minute)]
    cycles = 0
    a = time()
    if auto_ar:
        a_arima = auto_arima(initial_train, start_p=0, start_q=0, stepwise=True, suppress_warnings=True,
                             seasonal=True, stationary=True, trend='c', error_action='ignore', random_state=123)
    else:
        a_arima = ARIMA(order=(5, 0, 1))
    auto_t = time() - a

    # res_lst.append([0 for i in range(len(metric_lst))])
    #####################################################################################################

    a = time()
    updating_train = df['y'][int(0.5 * records_per_minute): records_per_minute]

    a_arima.fit(updating_train)
    if dump_model:
        dump(folder + '/%d.pkl'%(cycles), a_arima)

    b = time() - a
    # res_lst.append([0, *[0 for i in range(len(metric_lst))], 0, b])
    #####################################################################################################

    cycles = 0

    y_test = df['y'][records_per_minute * 1:]
    y_true, y_pred, test_score_lst = [], [], []
    while True:

        if int((cycles + 2) * cycle_records) > len(y_test): break

        one_minute_ahead_data = y_test[int(cycles * cycle_records): int((cycles + 2) * cycle_records)]
        y_true.append(one_minute_ahead_data)
        a = time()

        y_pred.append(a_arima.predict(n_periods=len(one_minute_ahead_data)))

        forecast_time = time() - a
        res = get_res(y_true[-1], y_pred[-1], metric_lst=metric_lst)

        # a half minute later, by the end of this cycle
        #####################################################################################################
        updating_train = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)].values
        a = time()
        a_arima.fit(updating_train)

        if dump_model:
            dump(folder + '/%d.pkl' % (cycles), a_arima)
        fitting_time = time() - a


        #####################################################################################################

        write(metric_lst, file=file_str,
              txt='min_%d\nGrid:%f\nFitting:%f\nPred:%f\n\n%r\n%s\n'%(cycles + 1,
                                                                      auto_t,
                                                                      fitting_time,
                                                                      forecast_time,
                                                                      res,'-'*80))

        res_lst.append(res)
        cycles += 1

    str_ = write(metric_lst, file=file_str, txt='\n' * 3,
                  data=res_lst, get_result=True)  # convert it to csv to get the mean of the runs
    with open(folder + '/00.txt', 'a') as myfile:
        myfile.write('%r'%(str_) + '\n')
