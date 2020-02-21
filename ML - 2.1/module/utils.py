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
             train_index=[6426,10427],
             test_index=[14389,17872],
             transform=None,
             drop_time=True,
             scale=True,
             return_y_scaler=False,
             path = '/Users/apple/Documents/ML_Project/ML - 2.1/Data/国际西班牙数据.csv'):
    # transform: can be one of [none, 'sin', 'cos', 'sin+cos', 
    #                           'ws*sin(wd)', 'ws*cos(wd)', 'ws*sin(wd)+ws*cos(wd)']
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
    
    print('get_data(hour_num={}, transform=\'{}\', drop_time={}, scale={})\n'\
        .format(hour_num, transform, drop_time, scale))
    print('Input space: ',X_train.columns,'\n')

    if scale & return_y_scaler:
        return X_train, X_test, Y_train, Y_test, Y_Scaler
    else:
        return X_train, X_test, Y_train, Y_test


from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE, CRPS
def model_test(Base, X_train, X_test, Y_train, Y_test, 
               n_estimators=500, learning_rate=0.01, Score=MLE,
               verbose=True, verbose_eval=100, 
               plot_predict=True, return_y_pred=False, 
               return_y_dists=False, return_mse=False):
    ngb = NGBRegressor(Base=Base, 
                       n_estimators=n_estimators,
                       verbose=verbose,
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
    if (return_y_pred) & (not(return_y_dists)):
        return pd.Series(Y_preds,index=Y_test.index)
    if (not(return_y_pred)) & (return_y_dists):
        return Y_dists
    if (return_y_pred) & (return_y_dists):
        return pd.Series(Y_preds,index=Y_test.index), Y_dists
    if return_mse:
        return test_MSE


def csv_to_heatmap(path, figsize=(15,8), vmin=0.01, vmax=0.04,
                   save_path='/Users/apple/Documents/ML_Project/ML - 2.1/result/plot/csv_to_heatmap.png'):
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
               save_path='/Users/apple/Documents/ML_Project/ML - 2.1/figure/'):
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
                path='/Users/apple/Documents/ML_Project/ML - 2.1/result/csv/',
                save_path='/Users/apple/Documents/ML_Project/ML - 2.1/result/'):
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



def set_func(func):
    num = [0]   # 闭包中外函数中的变量指向的引用不可变
    def call_func():
        func()
        num[0] += 1
        print("CV执行次数",num[0])
    return call_func
# 待测试方法
@set_func
def excute_time():
    pass