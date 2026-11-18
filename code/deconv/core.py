"""Image I/O and forward model. Kernel origin is explicitly floor(size/2)."""
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import fft

Array = NDArray[np.float64]

def image_array(value: np.ndarray) -> Array:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('Expected a finite nonempty 2D array')
    return a

def read_reference(path: Path, size: int = 256) -> Array:
    with Image.open(path) as im:
        im = im.convert('L')
        w, h = im.size
        if min(w, h) < size:
            raise ValueError(f'Image too small: {path}')
        x, y = (w-size)//2, (h-size)//2
        return np.asarray(im.crop((x, y, x+size, y+size)), dtype=np.float64)/255

def save_tiff(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array(a).astype(np.float32)).save(path, compression='raw')

def save_png(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(a, 0, 1)*255).astype(np.uint8)).save(path)

def gaussian_psf(size: int = 17, sigma: float = 1.9) -> Array:
    if size < 1 or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError('Positive size and finite positive sigma required')
    axis = np.arange(size, dtype=float)-size//2
    h = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(2*sigma**2))
    return h/h.sum()

def kernel_grid(kernel: np.ndarray, shape: tuple[int, int]) -> Array:
    k = image_array(kernel)
    if any(a > b for a, b in zip(k.shape, shape)):
        raise ValueError('Kernel larger than image')
    grid = np.zeros(shape)
    grid[:k.shape[0], :k.shape[1]] = k
    return np.roll(grid, (-(k.shape[0]//2), -(k.shape[1]//2)), axis=(0, 1))

def validate_psf(psf: np.ndarray, shape: tuple[int, int]) -> Array:
    h = image_array(psf)
    if np.any(h < 0) or not np.isclose(h.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError('PSF must be nonnegative and sum to one')
    return kernel_grid(h, shape)

def convolve(image: np.ndarray, psf: np.ndarray) -> Array:
    a = image_array(image)
    h = validate_psf(psf, a.shape)
    return fft.irfft2(fft.rfft2(a)*fft.rfft2(h), s=a.shape)

def degrade(reference: Array, psf: Array, noise: str, seed: int) -> tuple[Array, Array]:
    blurred = convolve(reference, psf)
    rng = np.random.default_rng(seed)
    if noise == 'gaussian':
        observed = np.maximum(blurred+rng.normal(0, 3/255, reference.shape), 0)
    elif noise == 'poisson':
        observed = rng.poisson(np.maximum(blurred, 0)*700)/700
    else:
        raise ValueError(f'Unknown noise: {noise}')
    return blurred, observed
