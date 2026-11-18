"""MSE/PSNR and local SSIM, population covariance, 11x11 Gaussian window."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .core import Array, image_array

def metrics(reference: Array, estimate: Array) -> dict[str, float]:
    a, b = image_array(reference), image_array(estimate)
    if a.shape != b.shape or min(a.shape) < 11:
        raise ValueError('Matching images at least 11x11 required')
    mse = float(np.mean((a-b)**2))
    def smooth(x: Array) -> Array:
        return gaussian_filter(x, sigma=1.5, radius=5, mode='reflect')
    ma, mb = smooth(a), smooth(b)
    va, vb = np.maximum(smooth(a*a)-ma*ma,0), np.maximum(smooth(b*b)-mb*mb,0)
    cov = smooth(a*b)-ma*mb
    s = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return dict(mse=mse, psnr=float(10*np.log10(1/mse)) if mse else float('inf'),
                ssim=float(s[5:-5,5:-5].mean()))
