from utils import plot_module1, plot_module2
from tqdm import tqdm
import numpy as np

# for day in tqdm(np.arange(1,2)):
#     plot_module1(year=2017, month=10, day=day, figsize=(8,13), 
#                  save_fig=False, close_fig=True)
for day in tqdm(np.arange(1,2)):
    plot_module2(year=2017, month=10, day=day, figsize=(8,10), 
                 save_fig=True, close_fig=True)