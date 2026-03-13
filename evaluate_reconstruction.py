import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_reconstruction(x_rec, x_orig):
    # Ensure range 0-1 or match types
    x_rec = x_rec.astype(np.float64)
    x_orig = x_orig.astype(np.float64)
    
    # Calculate metrics
    # data_range=1.0 assumes images are normalized 0-1
    psnr_val = psnr(x_orig, x_rec, data_range=1.0)
    ssim_val = ssim(x_orig, x_rec, data_range=1.0)
    
    print('\n=== Reconstruction Quality ===')
    print(f'PSNR = {psnr_val:.4f} dB')
    print(f'SSIM = {ssim_val:.4f}')
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_orig, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(x_rec, cmap='gray')
    plt.title('MS-BCS-SPL Reconstruction')
    plt.axis('off')
    
    plt.show()