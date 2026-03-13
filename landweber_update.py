import numpy as np

def landweber_update(coeffs, y_measurements, u, block_size_list):
    
    new_coeffs = list(coeffs)
    L = len(y_measurements)
    
    for l in range(L):
        # Access the tuple (H, V, D)
        current_details = list(new_coeffs[l+1]) 
        
        [H_img, V_img] = current_details[0].shape
        
        for d in range(3):
            subband = current_details[d]
            
            blocks = y_measurements[l][d]
            
            for info in blocks:
                r_start = info['i']
                c_start = info['j']
                h_blk = info['h']
                w_blk = info['w']
                
                # Boundary check
                r_end = min(r_start + h_blk, H_img)
                c_end = min(c_start + w_blk, V_img)
                
                # Extract patch
                patch = subband[r_start:r_end, c_start:c_end]
                x_old = patch.flatten(order='F')
                
                # Phi * x
                y_est = info['Phi'].forward(x_old)
                
                # Residual
                r_res = info['y'] - y_est
                
                # Phi' * r
                cor = info['Phi'].transpose(r_res)
                
                # Update
                x_new = x_old + u * cor
                
                # Reshape back
                patch_new = x_new.reshape((h_blk, w_blk), order='F')
                
                # Put back
                subband[r_start:r_end, c_start:c_end] = patch_new
            
            # Save updated subband
            current_details[d] = subband
        
        new_coeffs[l+1] = tuple(current_details)
        
    return new_coeffs