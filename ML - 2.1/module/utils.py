import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def plot_module1(year, month, day, figsize=(14,16), save_fig=False, close_fig=True):
    path = '/Users/apple/Documents/ML_Project/ML - 2.1/data/国际西班牙数据.csv'
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
        plt.savefig('/Users/apple/Documents/ML_Project/ML - 2.1/figure/{}\{}\{}.png'.format(year,month,day))
    if close_fig:
        plt.close()


def plot_module2(year, month, day, figsize=(14,14), save_fig=False, close_fig=True):
    path = '/Users/apple/Documents/ML_Project/ML - 2.1/data/国际西班牙数据.csv'
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
        plt.savefig('/Users/apple/Documents/ML_Project/ML - 2.1/figure/{}\{}\{}.png'.format(year,month,day))
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
def get_data(hour_num=0, 
             columns=['wind_speed', 'wind_direction', 'wind_power'], 
             train_index=[6426,10427],
             test_index=[14389,17872],
             transform=None,
             drop_time=True,
             scale=True,
             path = '/Users/apple/Documents/ML_Project/ML - 2.1/Data/国际西班牙数据.csv'):
    # transform: can be one of [none, 'sin', 'cos', 'sin+cos', 
    #                           'ws*sin(wd)', 'ws*cos(wd)', 'ws*sin(wd)+ws*cos(wd)']
    data= load_data(path, add_time=True, describe=False)

    if transform==None:
        pass
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
        columns = ['ws*cos(wd)']
    elif transform=='ws*sin(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)']
    elif transform=='ws*sin(wd)+ws*cos(wd)':
        data['ws*sin(wd)'] = data['wind_speed'] * np.sin(data['wind_direction'])
        data['ws*cos(wd)'] = data['wind_speed'] * np.cos(data['wind_direction'])
        data.drop(['wind_direction','wind_speed'], axis=1, inplace=True)
        columns = ['ws*sin(wd)', 'ws*cos(wd)']
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
        Y_train_index = Y_train.index
        Y_test_index = Y_test.index
        Y_train = pd.Series(Y_Scaler.fit_transform(Y_train.values.reshape(-1,1)).reshape(len(Y_train),), 
                            index=Y_train.index)
        Y_train.name = 'Y_train'
        Y_test = pd.Series(Y_Scaler.transform(Y_test.values.reshape(-1,1)).reshape(len(Y_test),),
                            index=Y_test.index)
        Y_test.name = 'Y_test'
    
    print('get_data(hour_num={}, transform=\'{}\', drop_time={}, scale={})\n'\
        .format(hour_num, transform, drop_time, scale))

    return X_train, X_test, Y_train, Y_test


from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE, CRPS
def model_test(Base, X_train, X_test, Y_train, Y_test, 
               n_estimators=500, verbose_eval=100, learning_rate=0.01, Score=MLE,
               plot_predict=True, return_y_pred=False):
    ngb = NGBRegressor(Base=Base, 
                       n_estimators=n_estimators,
                       verbose_eval=verbose_eval,
                       learning_rate=learning_rate,
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
        df = pd.concat([Y_test, pd.Series(Y_preds,index=Y_test.index)], axis=1)
        df.columns = ['test','pred']
        df.plot(figsize=(10,4), title='MSE:{}  NLL:{}'.
                format(round(test_MSE,4), round(test_NLL,4)))
    if return_y_pred:
        return pd.Series(Y_preds,index=Y_test.index)