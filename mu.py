import numpy as np

def mu(phi_obj):
    iter_count = 10
    N = phi_obj.N
    
    v = np.random.randn(N)
    v = v / np.linalg.norm(v)
    
    for k in range(iter_count):
        y = phi_obj.forward(v)
        w = phi_obj.transpose(y)
        
        nw = np.linalg.norm(w)
        if nw == 0:
            v = w
            break
        v = w / nw
        
    y = phi_obj.forward(v)
    w = phi_obj.transpose(y)
    
    lambda_max = np.real(np.dot(v.T, w))
    return lambda_max