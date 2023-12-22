import sys, os
import warnings;warnings.filterwarnings('ignore')

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils._utils import *
from utils.metrics import *
from utils.time_utils import *
from utils.HP_optim import *
from utils.scoring import *
from utils.conf import *
from time import time

metric_lst = [MAE_sklearn, MSE_sklearn, RMSE_sklearn, POCID]

def GRU_exp1(which_='all', fast_run=False, scaling=1, blocks= 5, inner= 5, trails= 5, exp_name='',
             save_csv_pred=False, save_model_h5=False, plt_loss_=False, ds=1):

    df = pd.read_csv('dataset/%s.csv'%str(ds), encoding='latin1')

    if which_ == 'all':
        minutes_start, minutes_end = 2, 5
    else:
        minutes_start, minutes_end = which_, which_

    if fast_run:
        exp_str = 'Fast ' + exp_name + ' ' + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT_Fast/Exp_1 %s_csv '%ds + exp_str)
        blocks = 3
        inner = 2
        trails = 1
        df = df[['time', 'y']][:200]
        records_per_second = 30
        epoches = 2
    else:
        exp_str = exp_name + ' ' + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT/Exp_1 %s_csv '%ds + exp_str)
        blocks = blocks
        inner = inner
        trails = trails
        df = df[['time', 'y']]
        records_per_second = 6000
        epoches = 50
    os.makedirs(folder)

    x_data, y_data, scaler = get_data_look_back(df['y'], scaling=scaling, window_size=10, verbose=1)
    for minutes_ in range(minutes_start, minutes_end + 1):
        train_len = records_per_second * minutes_
        x, x_test = x_data[:train_len], x_data[train_len:]
        y, y_test = y_data[:train_len], y_data[train_len:]
        X_train, y_train, X_valid, y_valid = train_test_split(x, y, train_ratio=0.7)

        if scaler is not None:
            y_train_inv = scaler.inverse_transform(y_train.values.reshape(-1,1))
            y_valid_inv = scaler.inverse_transform(y_valid.values.reshape(-1,1))
            y_test_inv  = scaler.inverse_transform(y_test.values.reshape(-1,1))

        for count in range(blocks):
            txt_str_ = ''
            models = []
            txt_str_ += 'min:' + str(minutes_) + '  Train&val:' + str(train_len) + ' Test:' + str(len(x_test)) + '\n'

            txt_str_ += '  X_train:%r X_valid;%r' % (X_train.shape, X_valid.shape) + '\n'
            setting_dict, str_ = get_best_architecture(X_train, y_train, X_valid, y_valid,
                                                       trails=trails, setup_dict=hyperopt_space_, epoches=epoches)
            train_score_lst, val_score_lst, test_score_lst, epoch_lst = [], [], [], []
            fitting_time, prediction_time = '  ', ''
            for i in range(1, 1 + inner):

                model = model_from_dict(setting_dict, X_train.shape)
                mod_str = 'min_%d trail_%d i_%d' % (minutes_, count, i) + '.h5'
                patience = 7
                es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
                checkpoint = ModelCheckpoint(folder+'/%s' % mod_str, monitor='val_loss', verbose=True, save_best_only=True,mode='min')

                st_time = time()
                hist = model.fit(X_train, y_train,
                                 validation_data=(X_valid, y_valid),
                                 epochs=setting_dict['nb_epochs'],
                                 batch_size=setting_dict['batch_size'],
                                 callbacks=[es, checkpoint]
                                 )
                fitting_time += '%-.4f' % ((time() - st_time) / 60.0) + ', '

                st_time = time()

                y_train_pred = model.predict(X_train)
                y_valid_pred = model.predict(X_valid)
                y_test_pred = model.predict(x_test)


                prediction_time += '%-.4f' % ((time() - st_time) / 60.0) + ', '

                if scaler is not None:
                    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
                    y_valid_pred_inv = scaler.inverse_transform(y_valid_pred)
                    y_test_pred_inv = scaler.inverse_transform(y_test_pred)
                else:
                    y_train_pred_inv = y_train_pred
                    y_valid_pred_inv = y_valid_pred
                    y_test_pred_inv = y_test_pred

                epoch_lst.append(len(hist.history['mae']))
                models.append(folder+'/'+mod_str)
                train_score_lst.append([*get_res(y_train_inv,y_train_pred_inv, metric_lst=metric_lst)])
                val_score_lst.append([*get_res(y_valid_inv, y_valid_pred_inv, metric_lst=metric_lst)])
                test_score_lst.append( [*get_res(y_test_inv, y_test_pred_inv, metric_lst=metric_lst)])

                if save_model_h5:
                    save_model(model, folder + '/' + mod_str)

                if save_csv_pred:
                    true_vs_pred_values = pd.DataFrame(
                                                        np.hstack([np.array(y_test_inv).reshape(-1, 1), np.array(y_test_pred_inv).reshape(-1, 1)]),
                                                        columns='y_test,y_pred'.split(','))
                    true_vs_pred_values.to_csv(folder + '/%d %d %d %s.csv' % (minutes_, count, i, mod_str[:-3]))

                if plt_loss_:
                    plt_loss(folder+'/%s_loss' % mod_str[:-3], hist, mae_)


            exp_ = 'min_%d trail_%d.txt' % (minutes_, count)
            file_str = folder + '/' + exp_
            write(metric_lst, file=file_str, txt=txt_str_ + '\n\n' + str_ + '\n\n')

            write(metric_lst, file=file_str, txt='Epoches:        ' + ', '.join([str(i) for i in epoch_lst]) + '   AVG:' + str( np.array(epoch_lst).mean()))
            write(metric_lst, file=file_str, txt='fitting_time: ' + fitting_time + ' => ' + str(np.array([float(str.strip(i)) for i in fitting_time.split(',')[:-1]]).mean()))
            write(metric_lst, file=file_str, txt='prediction_time ' + prediction_time + ' => ' + str(np.array([float(str.strip(i)) for i in prediction_time.split(',')[:-1]]).mean()) + '\n' * 3)

            write(metric_lst, file=file_str, txt='Training: ', data=train_score_lst)
            write(metric_lst, file=file_str, txt='Validation: ', data=val_score_lst)
            str_ = write(metric_lst, file=file_str, txt='Test: ', data=test_score_lst, models=models, get_result=True)
            with open(folder+'_zz.txt', 'a') as myfile:
                myfile.write(  '%r  '%str_ + exp_ + '\n' )

def GRU_exp2(minutes_start=[7, 10, 13], fast_run=False, scaling=1, blocks= 5, inner= 5, trails= 5, exp_name='',
             save_csv_pred=False, plt_loss_=False, ds=1):


    df = pd.read_csv('ds/%s.csv'%str(ds), encoding='latin1')

    if type(minutes_start) is int:
        minutes_start = [minutes_start]
    elif type(minutes_start) is list:
        pass

    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT_Fast/Exp_2 %s_csv '%ds + exp_str)
        blocks = 1
        inner = 2
        trails = 1
        df = df[['time', 'y']][:200]
        records_per_minute = 10
        epoches = 2
    else:
        exp_str = '' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT/Exp_2 %s_csv '%ds + exp_str)
        blocks = blocks
        inner = inner
        trails = trails
        df = df[['time', 'y']]
        records_per_minute = 6000
        epoches = 50
    os.makedirs(folder)

    for minutes in minutes_start:
        x_data, y_data, scaler = get_data_look_back(df['y'], scaling=scaling, window_size=10, verbose=0)
        train_len = records_per_minute * minutes
        x, x_test = x_data[:train_len], x_data[train_len:]
        y, y_test = y_data[:train_len], y_data[train_len:]
        X_train, y_train, X_valid, y_valid = train_test_split(x, y, train_ratio=0.7)

        if scaler is not None:
            y_train_inv = scaler.inverse_transform(y_train.values.reshape(-1, 1))
            y_valid_inv = scaler.inverse_transform(y_valid.values.reshape(-1, 1))
            y_test_inv = scaler.inverse_transform(y_test.values.reshape(-1, 1))

        for count in range(blocks):
            txt_str_ = ''
            models = []
            txt_str_ += 'min:' + str(minutes) + '  Train&val:' + str(train_len) + ' Test:' + str(len(x_test)) + '\n'

            txt_str_ += '  X_train:%r X_valid;%r' % (X_train.shape, X_valid.shape) + '\n'
            setting_dict, str_ = get_best_architecture(X_train, y_train, X_valid, y_valid,
                                                       trails=trails, setup_dict=hyperopt_space_, epoches=epoches)
            train_score_lst, val_score_lst, test_score_lst, epoch_lst = [], [], [], []
            fitting_time, prediction_time = '  ', ''

            for i in range(1, 1 + inner):
                model = model_from_dict(setting_dict, X_train.shape)
                mod_str = 'min_%d trail_%d i_%d'%(minutes, count, i) + '.h5'
                patience = 7
                es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
                checkpoint = ModelCheckpoint(folder+'/%s' % mod_str, monitor='val_loss', verbose=True, save_best_only=True, mode='min')

                st_time = time()
                hist = model.fit(X_train, y_train,
                                 validation_data=(X_valid, y_valid),
                                 epochs=setting_dict['nb_epochs'],
                                 batch_size=setting_dict['batch_size'],
                                 callbacks=[es, checkpoint]
                                 )
                fitting_time += '%-.4f' % ((time() - st_time) / 60.0) + ', '

                st_time = time()
                # y_pred = model.predict(x_test)
                forecasting_lst = []
                lst = list(x_test[0][:, 0])
                for i in range(x_test.shape[0]):
                    forecasting_lst.append(model.predict(np.array(lst).reshape(1, 10, 1)))
                    lst.append(forecasting_lst[-1][0][0])
                    lst = lst[1:]

                forecasting_lst = np.array(forecasting_lst).reshape(-1,1)
                prediction_time += '%-.4f' % ((time() - st_time) / 60.0) + ', '

                epoch_lst.append(len(hist.history['mae']))

                y_train_pred = model.predict(X_train)
                y_valid_pred = model.predict(X_valid)

                prediction_time += '%-.4f' % ((time() - st_time) / 60.0) + ', '

                if scaler is not None:
                    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
                    y_valid_pred_inv = scaler.inverse_transform(y_valid_pred)
                    y_test_pred_inv = scaler.inverse_transform(forecasting_lst)
                else:
                    y_train_pred_inv = y_train_pred
                    y_valid_pred_inv = y_valid_pred
                    y_test_pred_inv = forecasting_lst

                epoch_lst.append(len(hist.history['mae']))
                models.append(folder+'/'+mod_str)
                train_score_lst.append([*get_res(y_train_inv, y_train_pred_inv, metric_lst=metric_lst)])
                val_score_lst.append([*get_res(y_valid_inv, y_valid_pred_inv, metric_lst=metric_lst)])
                test_score_lst.append([*get_res(y_test_inv, y_test_pred_inv, metric_lst=metric_lst)])


                if plt_loss_:
                    plt_loss(folder+'/%s_loss' % mod_str[:-3], hist, mae_)

                if save_csv_pred:
                    true_vs_pred_values = pd.DataFrame(
                        np.hstack([np.array(y_test_inv).reshape(-1, 1), np.array(y_test_pred_inv).reshape(-1, 1)]),
                        columns='y_test,y_pred'.split(','))
                    true_vs_pred_values.to_csv(folder + '/%d %d %d %s.csv' % (minutes, count, i, mod_str[:-3]))

            exp_ = 'min_%d trail_%d.txt' % (minutes, count)
            file_str = folder +'/'+ exp_

            write(metric_lst, file=file_str, txt=txt_str_ + '\n\n' + str_ + '\n\n')

            write(metric_lst, file=file_str, txt='Epoches:        ' + ', '.join([str(i) for i in epoch_lst]) + '   AVG:' + str(
                np.array(epoch_lst).mean()))
            write(metric_lst, file=file_str, txt='fitting_time: ' + fitting_time + ' => ' + str(
                np.array([float(str.strip(i)) for i in fitting_time.split(',')[:-1]]).mean()))
            write(metric_lst, file=file_str, txt='prediction_time ' + prediction_time + ' => ' + str(
                np.array([float(str.strip(i)) for i in prediction_time.split(',')[:-1]]).mean()) + '\n' * 3)

            write(metric_lst, file=file_str, txt='Training: ', data=train_score_lst)
            write(metric_lst, file=file_str, txt='Validation: ', data=val_score_lst)
            str_ = write(metric_lst, file=file_str, txt='Test: ', data=test_score_lst, models=models, get_result=True)

            with open(folder+'_zz.txt', 'a') as myfile:
                myfile.write('%r  ' % str_ + exp_ + '\n')

def GRU_exp3(training_mins=None, alternating_window=None, updating_epoches=3, fast_run=False, scaling=1, blocks= 5, inner= 5, trails= 5,
             hyperopt_space=None, exp_name='', ds=1):

    if training_mins is None and alternating_window is None:
        combination = [ [0.5,0.25], [0.5,1.0], [1.0,0.5], [1.0,2.0]]
    else:
        combination = [[training_mins,alternating_window]]

    df = pd.read_csv('ds/%s.csv'%str(ds), encoding='latin1')
    if fast_run:
        exp_str = 'Fast ' + exp_name +  get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT_Fast/Exp_3 %s_csv '%ds + exp_str)
        blocks = 2
        inner = 2
        trails = 1
        updating_epoches = 1
        df = df[['time', 'y']][:200]
        records_per_minute = 60
        epoches = 2
    else:
        exp_str = exp_name +  get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT/Exp_3 %s_csv '%ds + exp_str)
        blocks = blocks
        inner = inner
        trails = trails
        df = df[['time', 'y']]
        records_per_minute = 6000
        epoches = 50
    os.makedirs(folder)

    if hyperopt_space is None:
        hyperopt_space=hyperopt_space_

    for training_mins, alternating_window in combination:
        train_len = int(records_per_minute * training_mins)
        x_data, y_data, scaler = get_data_look_back(df['y'], scaling=scaling, window_size=10, verbose=1)
        x, x_test = x_data[:train_len], x_data[train_len:]
        y, y_test = y_data[:train_len], y_data[train_len:]

        # start_index_for_cycle = train_len
        cycle_records = alternating_window*records_per_minute

        for count in range(blocks):
            txt_str_ = ''
            models = []
            txt_str_ += 'Training mins' + str(training_mins) + 'alternative_' + str(alternating_window) + '  Train&val:' + str(train_len) + ' Test:' + str(len(x_test)) + '\n'

            X_train, y_train, X_valid, y_valid = train_test_split(x, y, train_ratio=0.7)
            txt_str_ += '  X_train:%r X_valid;%r' % (X_train.shape, X_valid.shape) + '\n'
            setting_dict, str_ = get_best_architecture(X_train, y_train, X_valid, y_valid,
                                                       trails=trails, setup_dict=hyperopt_space,epoches=epoches)
            train_score_lst, val_score_lst, test_score_lst, epoch_lst = [], [], [], []

            y_true = []
            y_pred = []

            for i in range(1, 1 + inner):
                # Training period  =================================================================================================
                model = model_from_dict(setting_dict, X_train.shape)
                mod_str = 'min_%-.1f alt_%.1f trail_%d i_%d' % (training_mins, alternating_window, count, i) + '.h5'
                patience = 7
                es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
                checkpoint = ModelCheckpoint(folder+'/%s'% mod_str, monitor='val_loss', verbose=True, save_best_only=True,
                                             mode='min')
                hist = model.fit( X_train, y_train,
                                  validation_data=(X_valid, y_valid),
                                  epochs=setting_dict['nb_epochs'],
                                  batch_size=setting_dict['batch_size'],
                                  callbacks=[es, checkpoint])
                models.append(folder + '/' + mod_str)
                model = load_model(folder+'/%s'% mod_str, custom_objects={'POCID':POCID})
                cycles = 0
                input_lst = list(x_test[0][:, 0])

                while True:
                    try: # Forecasting ==========================================================================================
                        period_data = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)]
                        forecasting_lst = []
                        for i in range(len(period_data)):
                            forecasting_lst.append(model.predict(np.array(input_lst).reshape(1, 10, 1)))
                            input_lst.append(forecasting_lst[-1][0][0])
                            input_lst = input_lst[1:]
                        y_true.append(period_data)
                        y_pred.append(np.array(forecasting_lst).reshape(-1,1))
                        cycles += 1
                    except Exception as exc:
                        sys.exit()

                    if (cycles + 2) * cycle_records > len(y_test):
                        break


                    try: # Updating =============================================================================================
                        period_data = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)]
                        x_data, y_data, _ = get_data_look_back(period_data, scaling=False, window_size=10, verbose=1)
                        X_train, y_train, X_valid, y_valid = train_test_split(x_data, y_data, train_ratio=0.7)
                        model.fit(X_train, y_train,
                                  validation_data=(X_valid, y_valid),
                                  epochs=updating_epoches,
                                  batch_size=setting_dict['batch_size'],
                                  callbacks=[es, checkpoint]
                                  )
                        model = load_model(folder+'/%s'% mod_str, custom_objects={'POCID':POCID})
                        models.append(folder + '/' + mod_str)
                        input_lst = list(period_data[-10:].values) ##############
                        cycles += 1
                    except Exception as exc:
                        print('\n**** A002: Err:\n', traceback.format_exc())
                        sys.exit()

            for y_true_lst,y_pred_lst in zip(y_true,y_pred):
                if scaler is not None:
                    y_test_inv = scaler.inverse_transform(y_true_lst)
                    y_pred_inv = scaler.inverse_transform(y_pred_lst)
                else:
                    y_test_inv = y_true_lst
                    y_pred_inv = y_pred_lst

                test_score_lst.append([*get_res(y_test_inv, y_pred_inv, metric_lst=metric_lst)])

            exp_ = 'tr_%-.2f alt_%-.2f trail_%d.txt' % (training_mins, alternating_window, count)
            file_str = folder + '/' + exp_
            write(metric_lst, file=file_str, txt=txt_str_ + '\n\n' + str_ + '\n\n')
            test_score_lst = np.array(test_score_lst)
            str_ = write(metric_lst, file=file_str, txt='Test: ', data=test_score_lst, models=models, get_result=True)
            with open(folder + '_zz.txt', 'a') as myfile:
                myfile.write('%r  ' % str_ + exp_ + '\n')

def GRU_exp4(training_mins=0.33, alternating_window=0.5, updating_epoches=7, fast_run=False, blocks = 5, trails= 5,
             exp_name='', save_csv_pred=False, ds=1):
    def forecast_one_minute(initial_input, y_true, scaler, cycle, mod_str, save_csv_pred=False):
        model = load_model(mod_str, custom_objects={'POCID': POCID})
        forecasting_lst = []
        a = time()
        for _ in range(len(y_true)):
            forecasting_lst.append(model.predict(np.array(initial_input).reshape(1, 10, 1)))
            initial_input.append(forecasting_lst[-1][0][0])
            initial_input = initial_input[1:]
        forecast_time = time() - a
        forecasting_lst = np.array(forecasting_lst).reshape(-1, 1)
        if scaler is not None:
            y_test_inv = scaler.inverse_transform(y_true)
            y_pred_inv = scaler.inverse_transform(forecasting_lst)
        else:
            y_test_inv = y_true
            y_pred_inv = forecasting_lst

        res = get_res(y_test_inv, y_pred_inv, metric_lst=metric_lst)

        if save_csv_pred:
            true_vs_pred_values = pd.DataFrame(
                np.hstack([np.array(y_test_inv).reshape(-1, 1),
                           np.array(y_pred_inv).reshape(-1, 1)]),
                columns='y_test,y_pred'.split(','))
            true_vs_pred_values.to_csv(mod_str.split('/')[:-2] + 'csv' % (cycle, mod_str[:-3]))
        # write(metric_lst, file='txt/' + mod_str[:-3]+'.txt', txt='', data=res)

        return res, forecast_time

    df = pd.read_csv('ds/%s.csv'%str(ds), encoding='latin1')
    scaling = True
    if fast_run:
        exp_str = 'Fast ' + exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT_Fast/Exp_4 %s_csv '%ds + exp_str)
        blocks = 2
        trails = 1
        df = df[['time', 'y']][:100]
        records_per_minute = 30
        epoches = 2
    else:
        exp_str = exp_name + get_TimeStamp_str() + ' '
        folder = str.strip('OUTPUT/Exp_4 %s_csv '%ds + exp_str)
        blocks = blocks
        trails = trails
        df = df[['time', 'y']]
        records_per_minute = 6000
        epoches = 50
    os.makedirs(folder)

    train_len = int(records_per_minute * training_mins)

    x_data, y_data, scaler = get_data_look_back(df['y'], scaling=scaling, window_size=10, verbose=0)
    x, x_test = x_data[:train_len], x_data[train_len:]
    y, y_test = y_data[:train_len], y_data[train_len:]

    # start_index_for_cycle = train_len
    cycle_records = alternating_window*records_per_minute

    for count in range(blocks):
        txt_str_ = ''
        models = []
        extra_details = []
        txt_str_ += 'Training mins' + str(training_mins) + 'alt_' + str(alternating_window) + '  Train&val:' + str(train_len) + ' Test:' + str(len(x_test)) + '\n'

        X_train, y_train, X_valid, y_valid = train_test_split(x, y, train_ratio=0.7)
        txt_str_ += '  X_train:%r X_valid;%r' % (X_train.shape, X_valid.shape) + '\n'
        setting_dict, str_ = get_best_architecture(X_train, y_train, X_valid, y_valid,
                                                   trails=trails, setup_dict=hyperopt_space_,epoches=epoches)
        train_score_lst, val_score_lst, test_score_lst, epoch_lst = [], [], [], []
        cycles = 0

        # Training period  =================================================================================================
        model = model_from_dict(setting_dict, X_train.shape)
        mod_str = 'min_%-.1f alt_%.1f trail_%d' % (training_mins, alternating_window, count)
        patience = 7
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)
        checkpoint = ModelCheckpoint(folder+'/%s'% mod_str + '_mod_%d'%cycles  + '.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                     mode='min')
        a = time()
        res_lst = []
        a = time()
        hist = model.fit( X_train, y_train,
                          validation_data=(X_valid, y_valid),
                          epochs=setting_dict['nb_epochs'],
                          batch_size=setting_dict['batch_size'],
                          callbacks=[es, checkpoint],
                          verbose=0
                          )

        b = time() - a

        res_lst.append([0, 0 ,0, 0]) # results of first model
        extra_details.append([0, b, -1]) # results of first model
        models.append(folder + '/' + mod_str)

        input_lst = list(x_test[0][:, 0])
        timers = []

        while True:
            if int((cycles + 2) * cycle_records) > len(y_test):break

            # forecast one minute ahead
            one_minute_ahead_data = y_test[int(cycles * cycle_records): int((cycles + 2) * cycle_records)]
            mod = folder + '/%s' % mod_str + '_mod_%d' % (cycles) + '.h5'
            res, forecast_time = forecast_one_minute( initial_input = input_lst,
                                                      y_true = one_minute_ahead_data, scaler = scaler,
                                                      cycle=cycles, mod_str=mod,
                                                      save_csv_pred=save_csv_pred)

            # a half minute later, by the end of this cycle
            full_cycle_data = y_test[int(cycles * cycle_records): int((cycles + 1) * cycle_records)]
            x_data, y_data, _ = get_data_look_back(full_cycle_data, scaling=False, window_size=10, verbose=0)
            X_train, y_train, X_valid, y_valid = train_test_split(x_data, y_data, train_ratio=0.7)
            mod = folder+'/%s'% mod_str + '_mod_%d'%(cycles)+'.h5'
            model = load_model(mod, custom_objects={'POCID':POCID})
            a = time()
            mod = folder+'/%s'% mod_str + '_mod_%d'%(cycles+1)  + '.h5'
            checkpoint = ModelCheckpoint(mod, monitor='val_loss', verbose=0,
                                         save_best_only=True,
                                         mode='min')
            model.fit(X_train, y_train,
                      validation_data=(X_valid, y_valid),
                      epochs=updating_epoches,
                      batch_size=setting_dict['batch_size'],
                      callbacks=[es, checkpoint],
                      verbose=0
                      )
            fitting_time = time()-a
            models.append(mod)

            res_lst.append([*res])
            extra_details.append([forecast_time, fitting_time, cycles])

            input_lst = list(full_cycle_data[-10:].values)
            cycles += 1

        exp_ = mod_str
        res_lst = np.array(res_lst).reshape(-1, len(metric_lst))
        extra_details = np.array(extra_details)
        res_lst = np.hstack((  res_lst, extra_details  ))
        str_ = write(data=res_lst,
                     col=[*[f.__name__ for f in metric_lst], 'forecast_time', 'fitting_time', 'cycles' ], # convert it to csv to get the mean of the runs
                     metric_lst = metric_lst,
                     file=folder + '/' + exp_ + '.txt', txt='\n'*3,
                     models = models, get_result = True, ignore_first_raw=True)

        with open(folder + '_zz.txt', 'a') as myfile:
            myfile.write('%r  ' % str_ + exp_ + '\n')


































