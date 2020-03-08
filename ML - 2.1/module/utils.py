import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

work_path = '~/Documents/ML_Project'
# work_path = '/home/aistudio/work'

def load_data(path, add_time=True, describe=True):
    data = pd.read_csv(path)
    if describe:
        display(data.describe())
    if add_time:
        data['Date'] = data['Date'].apply(lambda data: data.split('/'))
        data['Year'] = data['Date'].apply(lambda data: int(data[2].split(' ')[0]))
        data['Month'] = data['Date'].apply(lambda data: int(data[1]))
        data['Day'] = data['Date'].apply(lambda data: int(data[0]))
        data['Hour'] = data['Date'].apply(lambda data: int(data[2].split(' ')[1].split(':')[0]))
        data.drop('Date', inplace=True, axis=1)
        data = data[['Year', 'Month', 'Day','Hour','wind_speed', 'wind_direction', 'wind_power']]
    return data


from sklearn.preprocessing import MinMaxScaler
def plot_module1(year, month, day, figsize=(14,16), save_fig=False, close_fig=True):
    path = work_path+'/ML - 2.1/data/国际西班牙数据.csv'
    df = load_data(path, add_time=True, describe=False)
    df['ws_sin(wd)'] = df['wind_speed']*np.sin(df['wind_direction'])
    df['ws_cos(wd)'] = df['wind_speed']*np.cos(df['wind_direction'])
    df['ws_wd'] = df['wind_speed']*df['wind_direction']
    df['ws(Scaled)_wd(Scaled)'] = \
    MinMaxScaler().fit_transform(df['wind_speed'].values.reshape(-1,1))*\
    MinMaxScaler().fit_transform(df['wind_direction'].values.reshape(-1,1))

    df['wind_power(Scaled)'] = \
    MinMaxScaler().fit_transform(df['wind_power'].values.reshape(-1,1))

    fig, axes = plt.subplots(4,2,figsize=figsize)
    fig.suptitle('Year\Month\Day: {}\{}\{}'.format(year,month,day), fontsize=15)
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_speed'].plot(ax=axes[0,0],title='wind_speed', ylim=[0,7])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_direction'].plot(ax=axes[1,0],title='wind_direction', ylim=[90,360])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_power'].plot(ax=axes[2,0], title='wind_power', ylim=[0,5000])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['ws_sin(wd)'].plot(ax=axes[0,1],title='wind_speed*sin(wind_direction)')
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['ws_cos(wd)'].plot(ax=axes[1,1],title='wind_speed * cos(wind_direction)')
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['ws_wd'].plot(ax=axes[2,1],title='wind_speed*wind_direction')

    pd.Series(MinMaxScaler().fit_transform\
    (df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]\
     ['wind_speed'].values.reshape(-1,1)).reshape(24,)).plot(ax=axes[3,0])

    pd.Series(MinMaxScaler().fit_transform\
    (df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]\
     ['wind_direction'].values.reshape(-1,1)).reshape(24,)).plot(ax=axes[3,0],title='wind_speed & wind_direction')
    if save_fig:
        plt.savefig(work_path+'/ML - 2.1/figure/{}\{}\{}.png'.format(year,month,day))
    if close_fig:
        plt.close()


def plot_module2(year, month, day, figsize=(14,14), save_fig=False, close_fig=True):
    path = work_path+'/ML - 2.1/data/国际西班牙数据.csv'
    df = load_data(path, add_time=True, describe=False)
    df['ws_sin(wd)'] = df['wind_speed']*np.sin(df['wind_direction'])
    df['ws_cos(wd)'] = df['wind_speed']*np.cos(df['wind_direction'])
    df['ws_wd'] = df['wind_speed']*df['wind_direction']
    df['ws(Scaled)_wd(Scaled)'] = \
    MinMaxScaler().fit_transform(df['wind_speed'].values.reshape(-1,1))*\
    MinMaxScaler().fit_transform(df['wind_direction'].values.reshape(-1,1))

    df['wind_power(Scaled)'] = \
    MinMaxScaler().fit_transform(df['wind_power'].values.reshape(-1,1))
    df = df
    fig, axes = plt.subplots(3,2,figsize=figsize)
    fig.suptitle('Year\Month\Day: {}\{}\{}'.format(year,month,day), fontsize=15)
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_speed'].plot(ax=axes[0,0],title='wind_speed', ylim=[0,7])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_direction'].plot(ax=axes[1,0],title='wind_direction', ylim=[90,360])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_power'].plot(ax=axes[2,0], title='wind_power', ylim=[0,5000])
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['wind_power(Scaled)'].plot(ax=axes[1,1],title='wind_power(Scaled)')
    df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]['ws_wd'].plot(ax=axes[2,1],title='wind_speed * wind_direction')
    
    pd.Series(MinMaxScaler().fit_transform\
              (df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]\
               ['wind_speed'].values.reshape(-1,1)).reshape(24,)).plot(ax=axes[0,1])

    pd.Series(MinMaxScaler().fit_transform\
              (df[(df['Year']==year) & (df['Month']==month) & (df['Day']==day)]\
               ['wind_direction'].values.reshape(-1,1)).reshape(24,)).plot(ax=axes[0,1],title='wind_speed(Scaled) & wind_direction(Scaled)')
    if save_fig:
        plt.savefig(work_path+'/ML - 2.1/figure/{}\{}\{}.png'.format(year,month,day))
    if close_fig:
        plt.close()

    
def Data_Extend_fun(Data, hour_num, columns):
    if hour_num==0:
        return Data
    elif hour_num>0:
        Extend_data = Data[hour_num:len(Data)].copy()
        Extend_data.index = np.arange(0,len(Extend_data)).tolist()
        for index in columns:
            for i in np.arange(1,hour_num+1):
                a = Data[index].iloc[(hour_num-i):-i].copy()
                a.index = np.arange(0,len(Extend_data)).tolist()
                Extend_data[index+'-{}'.format(i)] = a
        return Extend_data
    else:
        print('ERROR: hour num cannot be negative!')


from sklearn.preprocessing import MinMaxScaler
# [get_data]: function for get data from '国际西班牙数据'
#------- Input -------#
# hour_num: default 0, the 'l' in 't-l' feature
# train_index: default [6426,10426](len:4000), the index of train data
# test_index: default [14389,15389](len:1000), the index of test data
# transform: defaut None, can be one of these:
#           { None: Do nothing,
#            'sin': transform [wind_direction] to [sin(wind_direction)], 
#            'cos': transform [wind_direction] to [cos(wind_direction)], 
#            'sin+cos': transform [wind_direction] to [sin(wind_direction), cos(wind_direction)], 
#            'ws*sin(wd)': transform [wind_speed, wind_direction] to
#                          [wind_speed * sin(wind_direction)], 
#            'ws*cos(wd)': transform [wind_speed, wind_direction] to 
#                          [wind_speed * cos(wind_direction)], 
#            'ws*sin(wd)+ws*cos(wd)': transform [wind_speed, wind_direction] to 
#                          [wind_speed*sin(wind_direction), wind_speed*cos(wind_direction)]}
# drop_time: default True, drop time features: ['Year', 'Month', 'Day', 'Hour', 'Minute']
# scale: take MinMaxScaler on both X and Y data
# return_y_scaler: default False, return Y_train MinMaxScaler
# path: path for CSV file
#------- Output -------#
# Default: X_train, X_test, Y_train, Y_test
# scale & return_y_scaler: X_train, X_test, Y_train, Y_test, Y_Scaler
def get_data(hour_num=0, 
             train_index=[6426,10427],
             test_index=[14389,15390],
             transform=None,
             drop_time=True,
             scale=True,
             return_y_scaler=False,
             path = work_path+'/ML - 2.1/Data/国际西班牙数据.csv',
             verbose=True):
    data= load_data(path, add_time=True, describe=False)

    if transform==None:
        columns=['wind_speed', 'wind_direction', 'wind_power']
    elif transform=='sin':
        data['sin(wd)'] = np.sin(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'sin(wd)', 'wind_power']
    elif transform=='cos':
        data['cos(wd)'] = np.cos(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'cos(wd)', 'wind_power']
    elif transform=='sin+cos':
        data['sin(wd)'] = np.sin(data['wind_direction'])
        data['cos(wd)'] = np.cos(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'sin(wd)', 'cos(wd)', 'wind_power']
    elif transform=='ws*cos(wd)':
        data['ws*cos(wd)'] = data['wind_speed'] * np.cos(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*cos(wd)', 'wind_power']
    elif transform=='ws*sin(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)', 'wind_power']
    elif transform=='ws*sin(wd)+ws*cos(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data['ws*cos(wd)'] = data['wind_speed'] * np.cos(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)', 'ws*cos(wd)', 'wind_power']
    else:
        return print('ERROR: \'transform\' can only be [none, \'sin\', \'cos\', \'sin+cos\', \'ws*sin(wd)\', \'ws*cos(wd)\', \'ws*sin(wd)+ws*cos(wd)\']\n')
        
    Train = Data_Extend_fun(Data=data.iloc[train_index[0]:train_index[1]],
                            hour_num=hour_num,columns = columns)
    Test = Data_Extend_fun(Data=data.iloc[test_index[0]:test_index[1]],
                           hour_num=hour_num,columns = columns)

    X_train = Train.drop('wind_power', axis=1)
    X_test = Test.drop('wind_power', axis=1)
    Y_train = Train['wind_power']
    Y_test = Test['wind_power']
    
    if drop_time:
        X_train.drop(['Year', 'Month', 'Day','Hour'], axis=1, inplace=True)
        X_test.drop(['Year', 'Month', 'Day','Hour'], axis=1, inplace=True)
        
    if scale:
        X_Scaler = MinMaxScaler()
        X_columns = X_train.columns
        X_train = pd.DataFrame(X_Scaler.fit_transform(X_train), columns=X_columns)
        X_test = pd.DataFrame(X_Scaler.transform(X_test), columns=X_columns)

        Y_Scaler = MinMaxScaler()
        Y_train = pd.Series(Y_Scaler.fit_transform(Y_train.values.reshape(-1,1)).reshape(len(Y_train),), 
                            index=Y_train.index)
        Y_train.name = 'Y_train'
        Y_test = pd.Series(Y_Scaler.transform(Y_test.values.reshape(-1,1)).reshape(len(Y_test),),
                            index=Y_test.index)
        Y_test.name = 'Y_test'
    
    if verbose:
        print('get_data(hour_num={}, transform=\'{}\', drop_time={}, scale={})\n'\
            .format(hour_num, transform, drop_time, scale))
        print('Input space:',X_train.columns)
        print('train index:', train_index, 'train_len:', len(X_train))
        print('test index:', test_index, 'test_len:', len(X_test))

    if scale & return_y_scaler:
        return X_train, X_test, Y_train, Y_test, Y_Scaler
    else:
        return X_train, X_test, Y_train, Y_test


from sklearn.preprocessing import MinMaxScaler
# [get_data2]: function for get data from folder '相近8个地点2012年数据'
#------- Input -------#
# hour_num: default 0, the 'l' in 't-l' feature
# train_index: default [3001,7001](len:4000), the index of train data
# test_index: default [2000,3000](len:1000), the index of test data
# transform: defaut None, can be one of these:
#           { None: Do nothing,
#            'sin': transform [wind_direction] to [sin(wind_direction)], 
#            'cos': transform [wind_direction] to [cos(wind_direction)], 
#            'sin+cos': transform [wind_direction] to [sin(wind_direction), cos(wind_direction)], 
#            'ws*sin(wd)': transform [wind_speed, wind_direction] to
#                          [wind_speed * sin(wind_direction)], 
#            'ws*cos(wd)': transform [wind_speed, wind_direction] to 
#                          [wind_speed * cos(wind_direction)], 
#            'ws*sin(wd)+ws*cos(wd)': transform [wind_speed, wind_direction] to 
#                          [wind_speed*sin(wind_direction), wind_speed*cos(wind_direction)]}
# drop_time: default True, drop time features: ['Year', 'Month', 'Day', 'Hour', 'Minute']
# drop_else: default False, drop else features: ['air_temperature', 'surface_air_pressure', 'density']
# drop_minute: default False, if Ture, will only maintain the data will minute==0
# scale: take MinMaxScaler on both X and Y data
# return_y_scaler: default False, return Y_train MinMaxScaler
# path: path for CSV file
#------- Output -------#
# Default: X_train, X_test, Y_train, Y_test
# scale & return_y_scaler: X_train, X_test, Y_train, Y_test, Y_Scaler
def get_data2(hour_num=0, 
             train_index=[3001,7002],
             test_index=[2000,3001],
             transform=None,
             drop_time=True,
             drop_else=False,
             scale=True,
             return_y_scaler=False,
             path = work_path+'/ML - 2.1/Data/相近8个地点2012年数据/20738-2012.csv',
             verbose=True,
             drop_minute=False):
    # transform: can be one of [None, 'sin', 'cos', 'sin+cos', 
    #                           'ws*sin(wd)', 'ws*cos(wd)', 'ws*sin(wd)+ws*cos(wd)']
    
    data = pd.read_csv(path, skiprows=[0,1,2])
    columns_name = ['Year', 'Month', 'Day', 'Hour', 'Minute', 
                'wind_power','wind_direction', 'wind_speed',
                'air_temperature', 'surface_air_pressure',
                'density']
    data.columns = columns_name

    time_col = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    if drop_minute:
        data = data[data['Minute']==0]
        del data['Minute']
        time_col = ['Year', 'Month', 'Day', 'Hour']

    if transform==None:
        columns=['wind_speed', 'wind_direction', 'wind_power']
    elif transform=='sin':
        data['sin(wd)'] = np.sin(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'sin(wd)', 'wind_power']
    elif transform=='cos':
        data['cos(wd)'] = np.cos(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'cos(wd)', 'wind_power']
    elif transform=='sin+cos':
        data['sin(wd)'] = np.sin(data['wind_direction'])
        data['cos(wd)'] = np.cos(data['wind_direction'])
        data.drop(['wind_direction'], axis=1, inplace=True)
        columns = ['wind_speed', 'sin(wd)', 'cos(wd)', 'wind_power']
    elif transform=='ws*cos(wd)':
        data['ws*cos(wd)'] = data['wind_speed'] * np.cos(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*cos(wd)', 'wind_power']
    elif transform=='ws*sin(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)', 'wind_power']
    elif transform=='ws*sin(wd)+ws*cos(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data['ws*cos(wd)'] = data['wind_speed'] * np.cos(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)', 'ws*cos(wd)', 'wind_power']
    else:
        return print('ERROR: \'transform\' can only be [none, \'sin\', \'cos\', \'sin+cos\', \'ws*sin(wd)\', \'ws*cos(wd)\', \'ws*sin(wd)+ws*cos(wd)\']\n')
        
    if drop_else:
        data.drop(['air_temperature', 'surface_air_pressure', 'density'], axis=1, inplace=True)
    else:
        columns = columns + ['air_temperature', 'surface_air_pressure', 'density']

    Train = Data_Extend_fun(Data=data.iloc[train_index[0]:train_index[1]],
                            hour_num=hour_num,columns = columns)
    Test = Data_Extend_fun(Data=data.iloc[test_index[0]:test_index[1]],
                           hour_num=hour_num,columns = columns)

    X_train = Train.drop('wind_power', axis=1)
    X_test = Test.drop('wind_power', axis=1)
    Y_train = Train['wind_power']
    Y_test = Test['wind_power']
    
    if drop_time:
        X_train.drop(time_col, axis=1, inplace=True)
        X_test.drop(time_col, axis=1, inplace=True)
        
    if scale:
        X_Scaler = MinMaxScaler()
        X_columns = X_train.columns
        X_train = pd.DataFrame(X_Scaler.fit_transform(X_train), columns=X_columns)
        X_test = pd.DataFrame(X_Scaler.transform(X_test), columns=X_columns)

        Y_Scaler = MinMaxScaler()
        Y_train = pd.Series(Y_Scaler.fit_transform(Y_train.values.reshape(-1,1)).reshape(len(Y_train),), 
                            index=Y_train.index)
        Y_train.name = 'Y_train'
        Y_test = pd.Series(Y_Scaler.transform(Y_test.values.reshape(-1,1)).reshape(len(Y_test),),
                            index=Y_test.index)
        Y_test.name = 'Y_test'
    
    if verbose:
        print('get_data2(hour_num={}, transform=\'{}\', drop_time={}, drop_esle={}, scale={})\n'\
            .format(hour_num, transform, drop_time, drop_else, scale))
        print('Data:', path.split('/')[-2:],'\n')
        print('Input space:',X_train.columns)
        print('train index:', train_index, 'train_len:', len(X_train))
        print('test index:', test_index, 'test_len:', len(X_test))

    if scale & return_y_scaler:
        return X_train, X_test, Y_train, Y_test, Y_Scaler
    else:
        return X_train, X_test, Y_train, Y_test


from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE, CRPS
from ngboost.distns import Bernoulli, Normal, LogNormal
# [model_test]: model test for ngboost
#------- Process -------#
# X_train --> ngboost.fit_predict(base, X_train, Y_train)
#------- Input -------#
# 1. Base: base learner
# 2. ngboost_param: n_estimators, learning_rate, Score,
#                   verbose, verbose_eval
# 3. Data: X_train, X_test, Y_train, Y_test
# 4. Y_scaler: default None, if there is a Y_scaler, then the plot value will be inverse
#------- Output -------# [Optional]
# 1. default: None
# 2. Plot Predict Figure: plot_predict=True
# 3. Return Y Predict: return_y_pred
# 4. Return Y Distribution: return_y_dists
# 5. Return Y Predict & Distribution: return_y_pred, return_y_dists
# 6. Return Test MSE: return_mse
# Attention! The Y Predict will always be scaled not matter there is a Y_scaler or not
#------- Example -------#
# import sys
# sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
# from utils import *
# from ngboost.learners import *
# from sklearn.metrics import mean_squared_error
# import numpy as np
# from tqdm.notebook import tqdm as tqdm
# %config InlineBackend.figure_format='retina'
# X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
#                                             test_index=[14389, 15389],
#                                             drop_time=True, scale=True)
# n_readout=464
# n_components=30
# damping = 0.61758485
# weight_scaling = 0.94653868
# model_test(Base= esn_ridge_learner(
#                 n_readout=n_readout,
#                 n_components=n_components,
#                 damping = damping,
#                 weight_scaling = weight_scaling,
#                 alpha=0.01),
#             n_estimators=500, verbose_eval=100,
#             X_train=X_train, X_test=X_test,
#             Y_train=Y_train, Y_test=Y_test)
def model_test(Base, X_train, X_test, Y_train, Y_test, 
               n_estimators=500, learning_rate=0.01, Score=MLE, Dist=Normal,
               verbose=True, verbose_eval=100, 
               plot_predict=True, return_y_pred=False, 
               return_y_dists=False, return_mse=False,
               Y_scaler=None):
    ngb = NGBRegressor(Base=Base, 
                       n_estimators=n_estimators,
                       verbose=verbose,
                       verbose_eval=verbose_eval,
                       learning_rate=learning_rate,
                       Dist=Dist,
                       Score=Score)
    print(ngb,'\n')
    ngb.fit(X_train, Y_train)
    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test) # return norm method: mean std
    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('\nTest MSE', test_MSE)
    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL', test_NLL)

    if plot_predict:
        if Y_scaler is not None:
            df = pd.concat([pd.Series(Y_scaler.inverse_transform(Y_test.copy().values.reshape(-1,1)).reshape(-1,), 
                                        index=Y_test.index), 
                            pd.Series(Y_scaler.inverse_transform(np.array(Y_preds).reshape(-1, 1)).reshape(-1,),
                                        index=Y_test.index)], axis=1)
            df.columns = ['test','pred']
            df.plot(figsize=(10,4), title='MSE:{}  NLL:{}'.
                 format(round(test_MSE,4), round(test_NLL,4)))
        else:
            df = pd.concat([Y_test, pd.Series(Y_preds,index=Y_test.index)], axis=1)
            df.columns = ['test','pred']
            df.plot(figsize=(10,4), title='MSE:{}  NLL:{}'.
                    format(round(test_MSE,4), round(test_NLL,4)))
    if (return_y_pred) & (not(return_y_dists)):
        return pd.Series(Y_preds,index=Y_test.index)
    if (not(return_y_pred)) & (return_y_dists):
        return Y_dists
    if (return_y_pred) & (return_y_dists):
        return pd.Series(Y_preds,index=Y_test.index), Y_dists
    if return_mse:
        return test_MSE


from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE, CRPS
from simple_esn.simple_esn import SimpleESN
from ngboost.distns import Bernoulli, Normal, LogNormal
## [model_test_for_esn_base]: model test for esn based ngboost
#------- Process -------#
#     X_train 
# --> ESN.fit_transform(X_train) as new_X_train for ngboost
# --> ngboost.fit_predict(new_X_train, Y_train
#                         base='ridge' or 'svr' or 'kernel ridge')
#------- Input -------#
# 1. Base: sklearn learner
# 2. esn_param: dict={'n_readout', 'n_components', 
#                     'damping', 'weight_scaling'}
# 3. ngboost_param: n_estimators, learning_rate, Score, 
#                   verbose, verbose_eval
# 4. Data: X_train, X_test, Y_train, Y_test
#------- Output -------# [Optional]
# 1. default: None
# 2. Plot Predict Figure: plot_predict=True
# 3. Return Y Predict: return_y_pred
# 4. Return Y Distribution: return_y_dists
# 5. Return Y Predict & Distribution: return_y_pred, return_y_dists
# 6. Return Test MSE: return_mse
#------- Example -------#
# import sys
# sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
# from utils import *
# from ngboost.learners import *
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge
# %config InlineBackend.figure_format='retina'
# X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
#                                             test_index=[14389, 15389],
#                                             drop_time=True, scale=True)
# esn_param = {'n_readout': 464,
#              'n_components': 30,
#              'damping': 0.61758485,
#              'weight_scaling': 0.94653868}
# model_test_for_esn_base(Base=Ridge(alpha=0.01), 
#                esn_param = esn_param,
#                n_estimators=500, verbose_eval=100, Score=CRPS,
#                X_train=X_train, X_test=X_test,
#                Y_train=Y_train, Y_test=Y_test)
def model_test_for_esn_base(Base, esn_param, 
                   X_train, X_test, Y_train, Y_test, 
                   n_estimators=500, learning_rate=0.01, Score=MLE, Dist=Normal,
                   verbose=True, verbose_eval=100, 
                   plot_predict=True, return_y_pred=False, 
                   return_y_dists=False, return_mse=False):

    ESN = SimpleESN(n_readout=esn_param['n_readout'], 
                    n_components=esn_param['n_components'], 
                    damping=esn_param['damping'],
                    weight_scaling=esn_param['weight_scaling'], 
                    discard_steps=0, 
                    random_state=None)
    X_train = ESN.fit_transform(X_train)
    X_test = ESN.fit_transform(X_test)
    
    ngb = NGBRegressor(Base=Base, 
                       n_estimators=n_estimators,
                       verbose=verbose,
                       verbose_eval=verbose_eval,
                       learning_rate=learning_rate,
                       Dist=Dist,
                       Score=Score)
    print(ESN,'\n')
    print(ngb,'\n')
    ngb.fit(X_train, Y_train)
    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test) # return norm method: mean std
    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('\nTest MSE', test_MSE)
    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL', test_NLL)

    if plot_predict:
        df = pd.concat([Y_test, pd.Series(Y_preds,index=Y_test.index)], axis=1)
        df.columns = ['test','pred']
        df.plot(figsize=(10,4), title='MSE:{}  NLL:{}'.
                format(round(test_MSE,4), round(test_NLL,4)))
    if (return_y_pred) & (not(return_y_dists)):
        return pd.Series(Y_preds,index=Y_test.index)
    if (not(return_y_pred)) & (return_y_dists):
        return Y_dists
    if (return_y_pred) & (return_y_dists):
        return pd.Series(Y_preds,index=Y_test.index), Y_dists
    if return_mse:
        return test_MSE


def base_model_test(base, X_train, X_test, Y_train, Y_test, 
                   plot_predict=True, return_y_pred=False, 
                   return_mse=False, Y_scaler=None, verbose=True):

    base.fit(X_train, Y_train)
    Y_preds = base.predict(X_test)
    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    if verbose==True:
        print(base,'\n')
        print('\nTest MSE', test_MSE)

    if plot_predict:
        if Y_scaler is not None:
            df = pd.concat([pd.Series(Y_scaler.inverse_transform(Y_test.copy().values.reshape(-1,1)).reshape(-1,), 
                                        index=Y_test.index), 
                            pd.Series(Y_scaler.inverse_transform(np.array(Y_preds).reshape(-1, 1)).reshape(-1,),
                                        index=Y_test.index)], axis=1)
            df.columns = ['test','pred']
            df.plot(figsize=(10,4), title='MSE:{}'.
                 format(round(test_MSE,4)))
        else:
            df = pd.concat([Y_test, pd.Series(Y_preds,index=Y_test.index)], axis=1)
            df.columns = ['test','pred']
            df.plot(figsize=(10,4), title='MSE:{}'.
                    format(round(test_MSE,4)))
    if return_y_pred:
        return pd.Series(Y_preds,index=Y_test.index)
    if return_mse:
        return test_MSE


def csv_to_heatmap(path, figsize=(15,8), vmin=0.01, vmax=0.04,
                   save_path=work_path+'/ML - 2.1/result/plot/csv_to_heatmap.png'):
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path, index_col=0)
    elif path.split('.')[-1]=='xlsx':
        df = pd.read_excel(path, index_col=0)
        
    f, ax= plt.subplots(figsize=figsize,nrows=1)
    if vmax>0.03:
        sns.heatmap(df, ax=ax, vmin=vmin, vmax=vmax, annot=True, fmt='.3f')
    else:
        sns.heatmap(df, ax=ax, vmin=vmin, vmax=vmax, annot=True, fmt='.4f')
    plt.xticks(rotation=20) 
    plt.savefig(save_path, dpi=300)


def csv_to_MSE(path, test_len='auto', plot_fig=False, save_fig=False, add_model_title=True, figsize=(10,4),
               save_path=work_path+'/ML - 2.1/figure/'):
    df = pd.read_csv(path, index_col=0)
    if test_len=='auto':
        test_len=len(df)
    df = df.iloc[:test_len]
    MSE_dict={}
    for i in np.arange(1,len(df.columns)):
        Y_test = df['Y_test']
        Y_preds = df[df.columns[i]]
        test_MSE = mean_squared_error(Y_preds, Y_test)
        MSE_dict.update({df.columns[i]:test_MSE})
        if plot_fig:
            if add_model_title:
                title='Input: '+path.split('/')[-1].split('. ')[1].split('.')[0]+\
                '   MSE:{}'.format(round(test_MSE,4))
            else:
                title='  MSE:{}'.format(round(test_MSE,4))
            # test Mean Squared Error
            pd.concat([Y_test, Y_preds], axis=1).plot(figsize=figsize, title=title)    
            pd.Series(np.zeros(len(df)), index=df.index).plot(color='k')
            if save_fig:
                plt.savefig(save_path+df.columns[i])
                print('  Save figue in'+save_path)
    return MSE_dict


import os
import seaborn as sns
def csvs_to_MSE(test_len='auto', save_file=True, plot_figure=True, figsize=(15,8), vmin=0.01, vmax=0.04,
                path=work_path+'/ML - 2.1/result/csv/',
                save_path=work_path+'/ML - 2.1/result/'):
    folder = os.listdir(path)
    folder.remove('.DS_Store')
    MSE_df = pd.DataFrame()
    for file in folder:
        test_MSE = csv_to_MSE(path+file, test_len=test_len, plot_fig=False, save_fig=False)
        MSE_df = pd.concat([MSE_df, pd.Series(test_MSE, name=file.split('.csv')[0])], axis=1)
    MSE_df = MSE_df.sort_index(axis=1)
    if save_file:
        MSE_df.to_csv(save_path+'MSE with len {}.csv'.format(test_len))
    if plot_figure:
        csv_to_heatmap(path=save_path+'MSE with len {}.csv'.format(test_len),
                       figsize=figsize, vmin=vmin, vmax=vmax,
                       save_path=save_path+'plot/MSE with len {}.png'.format(test_len))
    return MSE_df


import datetime
def set_func(func):
    num = [0]   # 闭包中外函数中的变量指向的引用不可变
    def call_func():
        func()
        num[0] += 1
        print("CV执行次数",num[0], datetime.datetime.now())
    return call_func
# 待测试方法
@set_func
def excute_time():
    pass

from tqdm.notebook import tqdm as tqdm
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
def tree_model_plot1(best_param):
    X_train, X_test, Y_train, Y_test, Y_scaler = \
    get_data(hour_num=int(best_param['hour_num']),
             transform=best_param['transform'],
             drop_time=bool(best_param['drop_time']), 
             scale=True, return_y_scaler=True)
    Y_pred = DecisionTreeRegressor(max_depth=best_param['max_depth']).\
        fit(X_train, Y_train).predict(X_test)
    mse = mean_squared_error(Y_pred, Y_test)
    pd.concat([pd.Series(Y_pred, name='Pred', 
                         index=Y_test.index), Y_test], axis=1)\
    .plot(title='mse:'+str(round(mse, 8))+\
          '   depth:'+str(best_param['max_depth']),
          figsize=(12, 5))

def tree_model_plot2(best_param, drop_minute=False):
    X_train, X_test, Y_train, Y_test, Y_scaler = \
    get_data2(hour_num=int(best_param['hour_num']),
             transform=best_param['transform'],
             drop_time=bool(best_param['drop_time']),
             drop_else=bool(best_param['drop_else']),
             scale=True, return_y_scaler=True,
              drop_minute=drop_minute)
    Y_pred = DecisionTreeRegressor(max_depth=best_param['max_depth']).\
        fit(X_train, Y_train).predict(X_test)
    mse = mean_squared_error(Y_pred, Y_test)
    pd.concat([pd.Series(Y_pred, name='Pred', 
                         index=Y_test.index), Y_test], axis=1)\
    .plot(title='mse:'+str(round(mse, 8))+\
          '   depth:'+str(best_param['max_depth']),
          figsize=(12, 5))

def tree_heatmap1(mse_df):
    f, ax= plt.subplots(figsize=(15,18),nrows=3)
    sns.heatmap(mse_df.groupby(['transform','hour_num'])['mse'].mean().unstack(),
                ax=ax[0], vmax=0.008, annot=True, fmt='.5f')
    sns.heatmap(mse_df.groupby(['transform','max_depth'])['mse'].mean().unstack(),
                ax=ax[1], vmax=0.008, annot=True, fmt='.3f')
    sns.heatmap(mse_df.groupby(['transform','drop_time'])['mse'].mean().unstack(),
            ax=ax[2], vmax=0.008, annot=True, fmt='.5f')

def tree_heatmap2(mse_df):
    f, ax= plt.subplots(figsize=(19,24),nrows=4)
    sns.heatmap(mse_df.groupby(['transform','hour_num'])['mse'].mean().unstack(),
                ax=ax[0], vmax=0.00219, annot=True, fmt='.5f')
    sns.heatmap(mse_df.groupby(['transform','max_depth'])['mse'].mean().unstack(),
                ax=ax[1], vmax=0.00007, annot=True, fmt='.5f')
    sns.heatmap(mse_df.groupby(['transform','drop_time'])['mse'].mean().unstack(),
                ax=ax[2], vmax=0.00219, annot=True, fmt='.5f')
    sns.heatmap(mse_df.groupby(['transform','drop_else'])['mse'].mean().unstack(),
                ax=ax[3], vmax=0.00219, annot=True, fmt='.5f')

def tree_grid_search1(param_grid, plot=True, heatmap=True):
    mse_df = pd.DataFrame()
    for transform in tqdm(param_grid['transform']):
        for hour_num in param_grid['hour_num']:
            for drop_time in param_grid['drop_time']:
                X_train, X_test, Y_train, Y_test = \
                get_data(hour_num=hour_num, transform=transform, 
                         drop_time=drop_time, scale=True, verbose=False)
                for max_depth in param_grid['max_depth']:
                    Y_pred = DecisionTreeRegressor(max_depth=max_depth).\
                    fit(X_train, Y_train).predict(X_test)
                    mse = mean_squared_error(Y_pred, Y_test)
                    new_data = {'transform': transform,
                                'hour_num': hour_num,
                                'drop_time': drop_time,
                                'max_depth': max_depth,
                                'mse':mse}
                    mse_df = mse_df.append(new_data, ignore_index=True)  
    if plot:
        tree_model_plot1(dict(mse_df.iloc[mse_df['mse'].idxmin()]))

    mse_df['transform'].replace({None: 'None'}, inplace=True)
    print('best_param:\n', dict(mse_df.iloc[mse_df['mse'].idxmin()]),
      '\n\nbest_mse:', mse_df['mse'].min())

    if heatmap:
        tree_heatmap1(mse_df)
    return mse_df, dict(mse_df.iloc[mse_df['mse'].idxmin()])

def tree_grid_search2(param_grid, plot=True, heatmap=True):
    mse_df = pd.DataFrame()
    for transform in tqdm(param_grid['transform']):
        for hour_num in param_grid['hour_num']:
            for drop_time in param_grid['drop_time']:
                for drop_else in param_grid['drop_else']:
                    X_train, X_test, Y_train, Y_test = \
                    get_data2(hour_num=hour_num, transform=transform, 
                              drop_time=drop_time, drop_else=drop_else,
                              scale=True, verbose=False)
                    for max_depth in param_grid['max_depth']:
                        Y_pred = DecisionTreeRegressor(max_depth=max_depth).\
                        fit(X_train, Y_train).predict(X_test)
                        mse = mean_squared_error(Y_pred, Y_test)
                        new_data = {'transform': transform,
                                    'hour_num': hour_num,
                                    'drop_time': drop_time,
                                    'drop_else': drop_else,
                                    'max_depth': max_depth,
                                    'mse':mse}
                        mse_df = mse_df.append(new_data, ignore_index=True)  
    if plot:
        tree_model_plot2(dict(mse_df.iloc[mse_df['mse'].idxmin()]))  
            
    mse_df['transform'].replace({None: 'None'}, inplace=True)
    print('best_param:\n', dict(mse_df.iloc[mse_df['mse'].idxmin()]),
      '\n\nbest_mse:', mse_df['mse'].min())

    if heatmap:
        tree_heatmap2(mse_df)
    return mse_df, dict(mse_df.iloc[mse_df['mse'].idxmin()])