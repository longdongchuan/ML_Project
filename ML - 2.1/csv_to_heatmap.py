import os
os.chdir('/Users/apple/Documents/ML_Project/ML - 2.1')
from module.utils import csvs_to_MSE, csv_to_heatmap

csv_to_heatmap(path='/Users/apple/Documents/ML_Project/ML - 2.1/result/result.xlsx',
	       vmin=0.01,vmax=0.04)

csvs_to_MSE(path='/Users/apple/Documents/ML_Project/ML - 2.1/result/csv/', 
            save_file=True, plot_figure=True, test_len=1000,
            vmin=0.012, vmax=0.015)