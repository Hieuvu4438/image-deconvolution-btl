"""Measure PSF mismatch, alternate blur and boundary errors with locked parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from run_experiments import METHODS, write_csv, write_json, plt
from run_imagej import latest_run, tiff
from deconv.core import gaussian_psf, horizontal_motion_psf, save_tiff
from deconv.metrics import metrics
from deconv.solvers import solve


SCENARIOS = {
    'matched': 'Matched Gaussian',
    'narrow': 'Assume sigma = 1.4',
    'wide': 'Assume sigma = 2.4',
    'motion': 'Matched motion',
    'motion_as_gaussian': 'Motion as Gaussian',
    'zero': 'Zero as periodic boundary',
}
CROP = 16


def phantom():
    """Geometric intensity truth with points, lines, bars, edges and a flat field."""
    y, x = np.indices((256, 256))
    f = np.full((256, 256), .08)
    f[(x > 24) & (x < 105) & (y > 20) & (y < 114)] = .58
    f[(x - 181)**2 + (y - 65)**2 <= 37**2] = .76
    f[152:224, 31:36] = .9
    f[152:224, 41:46] = .9
    for width, start in [(2, 64), (4, 110), (8, 165)]:
        for index in range(4):
            left = start + 2 * index * width
            f[152:224, left:left + width] = .9
    f[125, [155, 161, 176, 188]] = 1
    return f


def augmented_inputs(ref, seed, periodic, h, motion):
    noise = np.random.default_rng(seed).normal(0, 3 / 255, ref.shape)
    motion_obs = np.maximum(convolve2d(ref, motion, mode='same', boundary='wrap') + noise, 0)
    zero_obs = np.maximum(convolve2d(ref, h, mode='same', boundary='fill') + noise, 0)
    return {
        'matched': (periodic, h),
        'narrow': (periodic, gaussian_psf(sigma=1.4)),
        'wide': (periodic, gaussian_psf(sigma=2.4)),
        'motion': (motion_obs, motion),
        'motion_as_gaussian': (motion_obs, h),
        'zero': (zero_obs, h),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve() if args.run else latest_run()
    out = run / 'sensitivity'
    out.mkdir()
    params = json.loads((run / 'config_frozen_before_test.json').read_text())['selected']['gaussian']
    methods = ['observed', *METHODS]
    motion = horizontal_motion_psf()
    rows = []

    for folder in sorted((run / 'test').glob('*_gaussian')):
        image_id = folder.name.split('_')[0]
        ref = tiff(folder / 'reference.tif')
        h = tiff(folder / 'psf.tif')
        h /= h.sum()
        seed = json.loads((folder / 'input.json').read_text())['seed']
        scenarios = augmented_inputs(ref, seed, tiff(folder / 'observed.tif'), h, motion)
        for case, (g, assumed) in scenarios.items():
            for method in methods:
                result = g if method == 'observed' else solve(g, assumed, method, **params[method])
                clipped = np.clip(result, 0, 1)
                for region, area in [('full', np.s_[:, :]), ('inner', np.s_[CROP:-CROP, CROP:-CROP])]:
                    rows.append(dict(id=image_id, scenario=case, method=method, region=region,
                                     **metrics(ref[area], clipped[area])))
    write_csv(out / 'metrics.csv', rows)

    summary = []
    for case in SCENARIOS:
        for region in ['full', 'inner']:
            baselines = {r['id']: r['psnr'] for r in rows
                         if (r['scenario'], r['region'], r['method']) == (case, region, 'observed')}
            for method in methods:
                subset = [r for r in rows if (r['scenario'], r['region'], r['method']) == (case, region, method)]
                gains = np.array([r['psnr'] - baselines[r['id']] for r in subset])
                summary.append(dict(scenario=case, region=region, method=method, n=len(subset),
                                    psnr_mean=np.mean([r['psnr'] for r in subset]),
                                    ssim_mean=np.mean([r['ssim'] for r in subset]),
                                    gain_mean=gains.mean(), gain_std=gains.std(ddof=1),
                                    images_improved=int(np.count_nonzero(gains > 0))))
    if len(rows) != 720 or any(r['n'] != 10 for r in summary):
        raise ValueError('Incomplete sensitivity experiment')
    write_csv(out / 'summary.csv', summary)
    write_json(out / 'protocol.json', dict(
        scenarios=SCENARIOS, lock=params, crop_each_side=CROP, motion_kernel=motion.tolist(),
        sigma_narrow=1.4, sigma_wide=2.4, gaussian_sigma=1.9,
        noise='same additive Gaussian realization per image across scenarios; negatives clipped to 0',
        phantom_seed=7351, summary='clipped [0,1]; paired gains over each scenario observation'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout='constrained')
    palette = ['#306b86', '#9e762e', '#764275', '#24715e']
    for ax, region in zip(axes, ['full', 'inner']):
        for method, color in zip(METHODS[1:], palette):
            values = [r['gain_mean'] for r in summary if r['method'] == method and r['region'] == region]
            ax.plot(values, range(len(SCENARIOS)), 'o-', color=color, label=method, alpha=.9)
        ax.set_yticks(range(len(SCENARIOS)), list(SCENARIOS.values()))
        ax.invert_yaxis()
        ax.axvline(0, color='.45', lw=1)
        ax.grid(axis='x', lw=.5, alpha=.5)
        ax.set_title('Full image' if region == 'full' else '16-pixel border excluded')
        ax.set_xlabel('Mean change in PSNR (dB)')
    axes[1].legend()
    fig.savefig(out / 'sensitivity.png', dpi=175)
    plt.close(fig)

    ref = phantom()
    h = gaussian_psf()
    noise = np.random.default_rng(7351).normal(0, 3 / 255, ref.shape)
    figure, axes = plt.subplots(2, 5, figsize=(12, 5.3), layout='constrained')
    profile, profile_axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, layout='constrained')
    scores = []
    for index, (label, kernel) in enumerate([('Gaussian', h), ('Motion', motion)]):
        name = 'gaussian' if index == 0 else 'motion'
        directory = out / f'phantom_{name}'
        g = np.maximum(convolve2d(ref, kernel, mode='same', boundary='wrap') + noise, 0)
        arrays = {'Reference': ref, 'Observed': g}
        save_tiff(directory / 'reference.tif', ref)
        save_tiff(directory / 'observed.tif', g)
        save_tiff(directory / 'psf.tif', kernel)
        scores.append(dict(psf=name, method='observed', **metrics(ref, np.clip(g, 0, 1))))
        for method in METHODS:
            result = solve(g, kernel, method, **params[method])
            save_tiff(directory / f'{method}.tif', result)
            scores.append(dict(psf=name, method=method, **metrics(ref, np.clip(result, 0, 1))))
            if method in ['TRIF', 'RL', 'RLTV']:
                arrays[method] = result
        for ax, (title, a) in zip(axes[index], arrays.items()):
            ax.imshow(a, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        axes[index, 0].text(-.08, .5, label, transform=axes[index, 0].transAxes,
                            rotation='vertical', ha='center', va='center')
        for title in ['Reference', 'Observed', 'TRIF', 'RLTV']:
            profile_axes[index].plot(np.arange(23, 105), arrays[title][180, 23:105],
                                     label=title, lw=1.1)
        profile_axes[index].set(title=label, ylabel='Intensity')
    profile_axes[-1].set_xlabel('Column index at row 180 (zero-based)')
    profile_axes[0].legend(ncol=4)
    figure.savefig(out / 'phantom.png', dpi=175)
    profile.savefig(out / 'profile.png', dpi=175)
    plt.close('all')
    write_csv(out / 'phantom_metrics.csv', scores)
    print(f'PASS: {len(rows)} sensitivity records and {len(scores)} phantom records. {out}')


if __name__ == '__main__':
    main()
