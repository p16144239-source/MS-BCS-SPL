import numpy as np
import cv2
from scipy.signal import convolve2d

def bn3(highpasses):

    eps_val = 1e-12
    iter_count = 3
    L = len(highpasses)
    psi_D_shr = list(highpasses)

    # --- Noise Estimation ---
    C_last = highpasses[0] 
    

    threshold_scale = 1
    sigma_n = (np.median(np.abs(C_last)) / 0.6745) * threshold_scale
    sigma_n_sq = sigma_n**2

    # --- Multiscale Processing ---
    for lev in range(L):
        C_complex = highpasses[lev] 
        C_abs = np.abs(C_complex)
        H, W, D = C_abs.shape
        
        # Prepare Parent
        if lev < L - 1:
            P_source = highpasses[lev + 1] 
            P = np.zeros((H, W, D))
            
            for k in range(D):
                parent_band = np.abs(P_source[:, :, k])
                P[:, :, k] = cv2.resize(parent_band, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            P = np.zeros((H, W, D))
        
        # Model 3 Processing
        C_hat = np.zeros_like(C_abs)
        window = np.ones((3, 3)) / 9.0
        
        for k in range(D):
            y1 = C_abs[:, :, k]  
            y2 = P[:, :, k]      
            
            var_y1 = convolve2d(y1**2, window, mode='same', boundary='fill', fillvalue=0)
            var_y2 = convolve2d(y2**2, window, mode='same', boundary='fill', fillvalue=0)
            
            sigma_1_sq = np.maximum(var_y1 - sigma_n_sq, eps_val)
            sigma_2_sq = np.maximum(var_y2 - sigma_n_sq, eps_val)
            
            C1 = (np.sqrt(3) * sigma_n_sq) / sigma_1_sq
            C2 = (np.sqrt(3) * sigma_n_sq) / sigma_2_sq
            
            sigma_1 = np.sqrt(sigma_1_sq)
            sigma_2 = np.sqrt(sigma_2_sq) # Unused in loop but calculated for clarity
            
            A_sq = (y1 / sigma_1)**2
            # Notice: B_sq uses sigma_2 from parent variance estimation
            B_sq = (y2 / np.sqrt(sigma_2_sq))**2 
            
            r = np.sqrt(A_sq + B_sq) + eps_val
            
            for _ in range(iter_count):
                denom1 = r + C1
                denom2 = r + C2
                
                term1 = A_sq / (denom1**2)
                term2 = B_sq / (denom2**2)
                
                g_val = term1 + term2 - 1
                dg_val = -2 * (A_sq / (denom1**3) + B_sq / (denom2**3))
                
                delta = g_val / (dg_val - eps_val)
                r = r - delta
                r = np.maximum(r, eps_val)
            
            denom_final = 1 + C1 / r
            w1 = y1 / denom_final
            
            C_hat[:, :, k] = w1
        
        original_phase = C_complex / (C_abs + eps_val)
        psi_D_shr[lev] = C_hat * original_phase

    return tuple(psi_D_shr)
