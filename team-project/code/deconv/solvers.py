"""TV1 equations 8,13,15-20; RLTV extends eq.21 using Sage et al. §3.9.

All operators use periodic boundaries. No output clipping is hidden in solvers.
Baseline uses full complex FFT and deliberately recomputes OTF each RL step.
Optimized uses real FFT and caches OTF; both implement the same mathematics.
"""
import numpy as np
from scipy import fft
from .core import Array, image_array, kernel_grid, validate_psf

def tikhonov_spectral(g: np.ndarray, h: np.ndarray, lam: float) -> np.ndarray:
    """Eq.13, also applicable to the scalar example H=0.1 in TV1."""
    return np.conj(h)*g/(abs(h)**2+lam)

def tv_divergence(f: Array, delta: float = .01) -> Array:
    """Forward gradient, negative adjoint divergence, periodic boundaries."""
    dx, dy = np.roll(f, -1, axis=0)-f, np.roll(f, -1, axis=1)-f
    norm = np.sqrt(dx*dx+dy*dy+delta*delta)
    px, py = dx/norm, dy/norm
    return px-np.roll(px, 1, axis=0)+py-np.roll(py, 1, axis=1)

def solve(g: np.ndarray, psf: np.ndarray, method: str, lam: float = 0.0,
          iterations: int = 20, optimized: bool = True, eps: float = 1e-12,
          nif_floor: float = 1e-4, delta: float = .01) -> Array:
    """Solve NIF/TRIF/RIF/RL/RLTV. RL initializes at the observed mean.

    RLTV: f_next = f * H^T(g/(Hf+eps)) / (1-lambda*div(grad f/|grad f|)).
    Reject nonpositive denominators rather than silently changing the update.
    """
    g = image_array(g)
    method = method.upper()
    if method not in {'NIF','TRIF','RIF','RL','RLTV'}:
        raise ValueError(f'Unknown solver {method}')
    if not np.isfinite([lam, eps, nif_floor, delta]).all() or lam < 0 or min(eps,nif_floor,delta) <= 0:
        raise ValueError('Invalid numerical parameters')
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    grid = validate_psf(psf, g.shape)
    forward = fft.rfft2 if optimized else fft.fft2
    def inverse(x: np.ndarray) -> Array:
        return fft.irfft2(x, s=g.shape) if optimized else fft.ifft2(x).real
    h = forward(grid)
    hc = h.conj()
    if method in {'NIF','TRIF','RIF'}:
        power = abs(h)**2
        if method == 'NIF':
            denominator = np.maximum(power, nif_floor)
        elif method == 'TRIF':
            if lam <= 0:
                raise ValueError('TRIF requires lambda > 0')
            return inverse(tikhonov_spectral(forward(g), h, lam))
        else:
            if lam <= 0:
                raise ValueError('RIF requires lambda > 0')
            lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
            denominator = power+lam*abs(forward(kernel_grid(lap,g.shape)))**2
        return inverse(hc*forward(g)/denominator)
    if np.any(g < 0):
        raise ValueError('RL and RLTV require nonnegative observations')
    f = np.full_like(g, g.mean())
    for _ in range(iterations):
        if not optimized:
            h = forward(grid)
            hc = h.conj()
        prediction = np.maximum(inverse(forward(f)*h), 0)
        correction = np.maximum(inverse(forward(g/(prediction+eps))*hc), 0)
        denominator = 1-lam*tv_divergence(f, delta) if method == 'RLTV' else 1.0
        if np.any(np.asarray(denominator) <= 0):
            raise ValueError('RLTV denominator nonpositive: reduce lambda')
        f = f*correction/denominator
        if not np.isfinite(f).all():
            raise FloatingPointError('Nonfinite iterate')
    return f

def naive_inverse_filter(g: Array, psf: Array, eps: float = 1e-4) -> Array:
    return solve(g, psf, 'NIF', nif_floor=eps)

def tikhonov_rif(g: Array, psf: Array, lam: float) -> Array:
    return solve(g, psf, 'TRIF', lam=lam)

def laplacian_rif(g: Array, psf: Array, lam: float) -> Array:
    return solve(g, psf, 'RIF', lam=lam)

def richardson_lucy(g: Array, psf: Array, num_iter: int = 20) -> Array:
    return solve(g, psf, 'RL', iterations=num_iter)

def richardson_lucy_tv(g: Array, psf: Array, num_iter: int = 20, lam: float = .001) -> Array:
    return solve(g, psf, 'RLTV', iterations=num_iter, lam=lam)
