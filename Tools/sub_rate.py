import numpy as np

def Sub_rate(L, subrate, S0, S1):

    W = np.zeros(L)
    for l in range(L):
        W[l] = 16**(L - (l + 1) + 1)

    fm = np.zeros(L, dtype=bool)
    S = np.zeros(L)
    S[0] = S1

    for iteration in range(3):
        be = 0.0
        cur = (1 / 4**L) * S0
        
        for l in range(L):
            coeff = 3 / (4**(L - (l + 1) + 1)) 
            
            if fm[l] or (l == 0 and S1 != 0):
                cur = cur + coeff * S[l]
            else:
                be = be + coeff * W[l]
        
        r = subrate - cur
        if be == 0:
            break
            
        Sp = r / be
        
        v = False
        for l in range(L):
            if not fm[l] and l != 0: 
                val = W[l] * Sp
                if val > 1.0:
                    S[l] = 1.0
                    fm[l] = True
                    v = True
                else:
                    S[l] = val
        
        if not v:
            break

    S = np.maximum(0, S)
    Subrate = S
    
    return Subrate
