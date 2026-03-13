import numpy as np
import cv2    # 影像前處理
import pywt    # DWT
import dtcwt    # DDWT
# from scipy.signal import wiener
import matplotlib.pyplot as plt

from SRMs import SRMs
from mu import mu
from wiener2 import wiener2
from sub_rate import Sub_rate
from bn3 import bn3
from landweber_update import landweber_update
from evaluate_reconstruction import evaluate_reconstruction

def main():
    img_path = "Lena.png"
    img = cv2.imread(img_path)
        
    # Convert to Gray and Normalize 0-1
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    
    # Parameters
    subrate_target = 0.3
    block_sizes = [64, 32, 16]
    L = 3
    wavelet_name = 'bior4.4'
    max_iter = 50
    tol = 1e-2     
    rng_seed = 0
    np.random.seed(rng_seed)
    
    # 2. Subrate
    S0 = 1
    S1 = 1
    target_subrates = Sub_rate(L, subrate_target, S0, S1)

    # 3. DWT 分解 x = Ω x
    C1 = pywt.wavedec2(x, wavelet_name, level=L)
    A0 = C1[0]

    # 4. 採樣 y = Φ x
    y_measurements = []
    
    # pywt: details_list[0] is Level L (Coarse), [1] is Level L-1
    details_list = C1[1:] 

    first_phi = None
    
    for l_idx, details in enumerate(details_list):
        blk = block_sizes[l_idx]
        current_subrate = target_subrates[l_idx]
        
        level_y = []
        
        # Details is (H, V, D)
        for d_idx, band in enumerate(details):
            H_band, W_band = band.shape
            
            band_blocks = []
            
            # Loop through blocks
            for i in range(0, H_band, blk):
                for j in range(0, W_band, blk):
                    hi = min(i + blk, H_band)
                    wj = min(j + blk, W_band)
                    
                    real_h = hi - i
                    real_w = wj - j
                    
                    patch = band[i:hi, j:wj]
                    vec = patch.flatten(order='F')
                    n = real_h * real_w
                    
                    M = max(1, int(round(current_subrate * n)))
                    
                    # Phi
                    phi_obj = SRMs(n, M) 
                    
                    y_meas = phi_obj.forward(vec)
                    
                    info = {
                        'y': y_meas,
                        'Phi': phi_obj,
                        'i': i, 'j': j,
                        'h': real_h, 'w': real_w
                    }
                    band_blocks.append(info)
                    
                    if first_phi is None:
                        first_phi = phi_obj
            
            level_y.append(band_blocks)
        y_measurements.append(level_y)

    # 5. 初始化估計 x̃(0) = Φᵀ y
    det_est = []
    for l_idx, details in enumerate(details_list):
        level_est = []
        for d_idx in range(3):
            H_band, W_band = details[d_idx].shape
            est_band = np.zeros((H_band, W_band))
            
            blocks = y_measurements[l_idx][d_idx]
            for info in blocks:
                vec_est = info['Phi'].transpose(info['y'])
                patch = vec_est.reshape((info['h'], info['w']), order='F')
                est_band[info['i']:info['i']+info['h'], info['j']:info['j']+info['w']] = patch
            
            level_est.append(est_band)
        det_est.append(tuple(level_est))
    
    coeffs_est = [A0] + det_est
    
    # 6. Main loop
    lambda_max = mu(first_phi)
    u_step = (1.0 / lambda_max) 

    D_history = []
    transform = dtcwt.Transform2d()
    current_coeffs = coeffs_est

    for it in range(max_iter):
        # A. Wiener Filtering
        x_curr = pywt.waverec2(current_coeffs, wavelet_name)
        x_hat = wiener2(x_curr, (3, 3))
        
        # B. Landweber Step 1
        coeffs_hat = pywt.wavedec2(x_hat, wavelet_name, level=L)
        coeffs_LW1 = landweber_update(coeffs_hat, y_measurements, u_step, block_sizes)
        x_LW1 = pywt.waverec2(coeffs_LW1, wavelet_name)
        
        # C. DDWT
        t = transform.forward(x_LW1, nlevels=L)
        psi_D_shr = bn3(t.highpasses) # bn3
        t_shr = dtcwt.Pyramid(t.lowpass, psi_D_shr)
        x_sparse = transform.inverse(t_shr)
        x_sparse = np.clip(x_sparse, 0, 1)
        
        # D. Landweber Step 2
        coeffs_bar = pywt.wavedec2(x_sparse, wavelet_name, level=L)
        coeffs_next = landweber_update(coeffs_bar, y_measurements, u_step, block_sizes)
        
        # 強制鎖定低頻 A0，防止畫面變淡
        coeffs_next[0] = A0

        # E. Convergence Check
        def flatten_coeffs(c):
            arrs = [c[0].flatten()]
            for d in c[1:]:
                for band in d:
                    arrs.append(band.flatten())
            return np.concatenate(arrs)
            
        prev_vec = flatten_coeffs(coeffs_LW1)
        curr_vec = flatten_coeffs(coeffs_next)
        
        diff = np.linalg.norm(curr_vec - prev_vec)
        D_history.append(diff)
        
        print(f'Iter {it+1}: D={diff:.5f}')
        
        if it > 5 and abs(D_history[-1] - D_history[-2]) < tol:
            print("Converged.")
            x_final = pywt.waverec2(coeffs_next, wavelet_name)
            break
            
        current_coeffs = coeffs_next
        x_final = pywt.waverec2(coeffs_next, wavelet_name)

    evaluate_reconstruction(x_final, x)

main()