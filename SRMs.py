import numpy as np
from scipy.fftpack import dct, idct

class SRMs:
    def __init__(self, N, M, rng_seed=None):
        if rng_seed is not None:
            np.random.seed(rng_seed)
        
        self.N = N
        self.M = M
        
        # 1. Random Sign Flip
        self.signs = np.random.randint(0, 2, N) * 2 - 1
        
        # 2. Random Permutation
        self.perm_idx = np.random.permutation(N)
        
        # Inverse permutation for transpose
        self.inv_perm_idx = np.argsort(self.perm_idx)

    def forward(self, x):
        #  x -> D -> DCT -> P -> R -> y
        x = x.flatten()
        
        x1 = x * self.signs
        x2 = dct(x1, type=2, norm='ortho') 
        x3 = x2[self.perm_idx]
        y = x3[:self.M]
        return y

    def transpose(self, y):
        # y -> R' -> P' -> IDCT -> D' -> x
        x3 = np.zeros(self.N)
        x3[:self.M] = y.flatten()
        x2 = np.zeros(self.N)
        x2 = x3[self.inv_perm_idx]
        
        x1 = idct(x2, type=2, norm='ortho')
        x = x1 * self.signs
        return x