import numpy as np
from scipy.ndimage import uniform_filter

def wiener2(img, mysize=None, noise=None):
    if mysize is None:
        mysize = [3, 3]
    h, w = mysize
    
    # 估計局部均值與變異數
    local_mean = uniform_filter(img, (h, w))
    local_sqr_mean = uniform_filter(img**2, (h, w))
    local_var = local_sqr_mean - local_mean**2
    
    # 如果沒給噪聲功率，就用所有局部變異數的平均來估計
    if noise is None:
        noise = np.mean(local_var)
        
    # Wiener 公式
    res = img - local_mean
    res = res * (1 - noise / (local_var + 1e-10)) # 1e-10 避免除以 0
    res = res + local_mean
    
    gain = np.maximum(0, local_var - noise) / (local_var + 1e-10)
    res = local_mean + gain * (img - local_mean)
    
    return res
