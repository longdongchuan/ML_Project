% Demo script

%加载数据
data = load('MGtimeseries.mat');   
data = data.MGtimeseries;
inputData = cell2mat(data(1:end-1))'; 
targetData = cell2mat(data(2:end))';

%需要washout的数据个数
washout = 100;

%train和test数据集
trlen = 2000; tslen = 2000; 
trX{1} = inputData(1:trlen);
tsX{1} = inputData(trlen+1:trlen+tslen);
% Remove initial points from target!
trY = targetData(1+washout:trlen);
tsY = targetData(trlen+1+washout:trlen+tslen);

%esn参数设置
esn = ESN(50, 'leakRate', 0.3, 'spectralRadius', 0.5, 'regularization', 1e-8);

%esn训练
esn.train(trX, trY, washout);

%储蓄池计算的输出和输入(trX)合并
output=esn.internalState;
