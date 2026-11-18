"""Periodic deconvolution. RLTV uses the smooth isotropic TV functional.

All operators use periodic boundaries. No output clipping is hidden in solvers.
Baseline uses full complex FFT and deliberately recomputes OTF each RL step.
Optimized uses real FFT and caches OTF; both implement the same mathematics.
"""
import numpy as np
from scipy import fft
from .core import Array, image_array, validate_psf

def tikhonov_spectral(
    observed_spectrum: np.ndarray, otf: np.ndarray, lam: float
) -> np.ndarray:
    """Apply identity-penalty Tikhonov filtering to the supplied spectrum."""
    return otf.conj() * observed_spectrum / (abs(otf)**2 + lam)

def tv_divergence(estimate: Array, delta: float = .01) -> Array:
    """Forward gradient, negative adjoint divergence, periodic boundaries."""
    row_gradient = np.roll(estimate, -1, axis=0) - estimate
    col_gradient = np.roll(estimate, -1, axis=1) - estimate
    magnitude = np.sqrt(row_gradient**2 + col_gradient**2 + delta**2)
    row_flux, col_flux = row_gradient / magnitude, col_gradient / magnitude
    row_divergence = row_flux - np.roll(row_flux, 1, axis=0)
    col_divergence = col_flux - np.roll(col_flux, 1, axis=1)
    return row_divergence + col_divergence

def solve(g: np.ndarray, psf: np.ndarray, method: str, lam: float = 0.0,
          iterations: int = 20, optimized: bool = True, eps: float = 1e-12,
          nif_floor: float = 1e-4, delta: float = .01, *,
          imagej_rl: bool = False) -> Array:
    """Solve NIF/TRIF/RIF/RL/RLTV. RL initializes at the observed mean.

    RLTV: f_next = f * H^T(g/(Hf+eps)) / (1-lambda*div(grad f/|grad f|)).
    imagej_rl matches the plugin's RL initialization and division rule for
    numerical cross-validation. It applies only to plain RL.
    """
    observation = image_array(g)
    method = method.upper()
    if method not in {'NIF', 'TRIF', 'RIF', 'RL', 'RLTV'}:
        raise ValueError(f'Unknown solver {method}')
    if imagej_rl and method != 'RL':
        raise ValueError('The ImageJ update option applies only to RL')
    if not np.isfinite([lam, eps, nif_floor, delta]).all() or lam < 0:
        raise ValueError('Invalid numerical parameters')
    if min(eps, nif_floor, delta) <= 0:
        raise ValueError('Invalid numerical parameters')
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    image_shape = observation.shape
    psf_grid = validate_psf(psf, image_shape)
    forward_fft = fft.rfft2 if optimized else fft.fft2

    def inverse_fft(spectrum: np.ndarray) -> Array:
        return fft.irfft2(spectrum, s=image_shape) if optimized else fft.ifft2(spectrum).real

    otf = forward_fft(psf_grid)
    adjoint_otf = otf.conj()
    if method in {'NIF', 'TRIF', 'RIF'}:
        if method != 'NIF' and lam <= 0:
            raise ValueError(f'{method} requires lambda > 0')
        observed_spectrum = forward_fft(observation)
        if method == 'TRIF':
            return inverse_fft(tikhonov_spectral(observed_spectrum, otf, lam))

        blur_power = abs(otf)**2
        if method == 'NIF':
            denominator = np.maximum(blur_power, nif_floor)
        else:
            row_frequency = fft.fftfreq(image_shape[0])[:, None]
            col_frequency = (fft.rfftfreq if optimized else fft.fftfreq)(image_shape[1])
            laplacian = -4 * (np.sin(np.pi * row_frequency)**2 + np.sin(np.pi * col_frequency)**2)
            denominator = blur_power + lam * laplacian**2
        return inverse_fft(adjoint_otf * observed_spectrum / denominator)

    if np.any(observation < 0):
        raise ValueError('RL and RLTV require nonnegative observations')
    estimate = observation.copy() if imagej_rl else np.full_like(observation, observation.mean())
    for _ in range(iterations):
        if not optimized:
            otf = forward_fft(psf_grid)
            adjoint_otf = otf.conj()
        prediction = inverse_fft(forward_fft(estimate) * otf)
        if imagej_rl:
            ratio = np.divide(observation, prediction, out=np.zeros_like(observation),
                              where=prediction >= eps)
        else:
            ratio = observation / (np.maximum(prediction, 0) + eps)
        correction = inverse_fft(forward_fft(ratio) * adjoint_otf)
        if not imagej_rl:
            correction = np.maximum(correction, 0)
        updated = estimate * correction
        if method == 'RLTV' and lam > 0:
            denominator = 1 - lam * tv_divergence(estimate, delta)
            if np.any(denominator <= 0):
                raise ValueError('RLTV denominator nonpositive: reduce lambda')
            updated /= denominator
        if not np.isfinite(updated).all():
            raise FloatingPointError('Nonfinite iterate')
        estimate = updated
    return estimate

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
