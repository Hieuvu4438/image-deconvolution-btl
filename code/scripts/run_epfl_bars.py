"""Evaluate a 2-D axial-sum projection of EPFL's public synthetic hollow bars."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageSequence
from scipy import fft

from run_experiments import ROOT, METHODS, plt, write_csv, write_json
from run_imagej import latest_run, tiff
from deconv.core import convolve, degrade, save_tiff
from deconv.metrics import metrics
from deconv.solvers import solve


SOURCE = 'https://bigwww.epfl.ch/deconvolution/bars/'
STACKS = {
    'Bars-stack': '1fbfa441a217771f98df889cfcb1e6b3867835c09cbde29c23fc6b4ce71cdee8',
    'PSF-Bars-stack': '63770fe4a64150f4227f98d1c7fa1b0cb637f03c5c24f16d78ce2fb28b47eb47',
}
SEEDS = {'gaussian': 735101, 'poisson': 735102}


def read_stack(name):
    cache = ROOT / 'data/epfl-bars'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'{name}.zip'
    url = SOURCE + archive.name
    data = archive.read_bytes() if archive.is_file() else None
    if data is None:
        with urlopen(url, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != STACKS[name]:
        raise ValueError(f'Stack archive checksum mismatch: {url}')
    if not archive.is_file():
        temporary = archive.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(archive)
    with ZipFile(BytesIO(data)) as zipped:
        with Image.open(BytesIO(zipped.read(f'{name}.tif'))) as image:
            stack = np.stack([np.array(frame) for frame in ImageSequence.Iterator(image)])
    if stack.shape != (128, 256, 256) or stack.dtype.kind != 'u' or stack.dtype.itemsize != 2:
        raise ValueError(f'Unexpected TIFF layout: {name}, {stack.shape}, {stack.dtype}')
    return stack


def project_blur(reference_volume, psf_volume, reference_peak, psf_sum):
    """Numerically check axial summation commutes with circular 3-D convolution."""
    signal = fft.rfftn(reference_volume.astype(np.float32) / reference_peak)
    centered_psf = psf_volume.astype(np.float32) / psf_sum
    signal *= fft.rfftn(fft.ifftshift(centered_psf))
    blurred_volume = fft.irfftn(signal, s=reference_volume.shape, axes=(0, 1, 2))
    return blurred_volume.sum(axis=0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve() if args.run else latest_run()
    output = run / 'epfl-bars'
    output.mkdir(exist_ok=False)
    frozen = run / 'config_frozen_before_test.json'
    config = json.loads(frozen.read_text())
    (output / frozen.name).write_bytes(frozen.read_bytes())

    volume, psf_volume = (read_stack(name) for name in STACKS)
    projection = volume.sum(axis=0, dtype=np.float64)
    reference_peak = float(projection.max())
    reference = (projection / reference_peak).astype(np.float32).astype(np.float64)
    psf = psf_volume.sum(axis=0, dtype=np.float64)
    psf_sum = float(psf.sum())
    psf_file = (psf / psf_sum).astype(np.float32)
    psf = psf_file.astype(np.float64)
    psf /= psf.sum()
    centered = np.unravel_index(psf.argmax(), psf.shape)
    if centered != (128, 128):
        raise ValueError(f'Unexpected PSF center {centered}')
    blurred_3d = project_blur(volume, psf_volume, np.float32(reference_peak), np.float32(psf_sum))
    projection_error = float(abs(blurred_3d - convolve(reference, psf)).max())
    if projection_error >= 1e-6:
        raise ValueError(f'Projection identity failed: {projection_error}')
    psf_mass_17 = float(psf[120:137, 120:137].sum())
    manifest = dict(
        source=SOURCE, archives={f'{k}.zip': v for k, v in STACKS.items()},
        shape_zyx=list(volume.shape), projection='sum over z, then fixed scaling',
        reference_peak=reference_peak, psf_sum=psf_sum, psf_center=[128, 128],
        psf_mass_17=psf_mass_17, projection_max_abs_error=projection_error,
        boundary='periodic in all three axes for identity test', seeds=SEEDS,
        synthetic_observation='Gaussian sigma 3/255 or photon peak 700, generated for this study')
    write_json(output / 'manifest.json', manifest)

    methods = ['observed', *METHODS]
    scores = []
    for noise, seed in SEEDS.items():
        folder = output / 'test' / f'epflbars_{noise}'
        blurred, observed = degrade(reference, psf, noise, seed)
        observed = observed.astype(np.float32).astype(np.float64)
        for name, array in [('reference', reference), ('psf', psf_file),
                            ('observed', observed), ('blurred', blurred)]:
            save_tiff(folder / f'{name}.tif', array)
        for method in methods:
            result = observed if method == 'observed' else solve(
                observed, psf, method, **config['selected'][noise][method])
            if method != 'observed':
                save_tiff(folder / f'{method}.tif', result)
                result = tiff(folder / f'{method}.tif')
            for policy in ('raw', 'clip01'):
                image = result if policy == 'raw' else np.clip(result, 0, 1)
                scores.append(dict(noise=noise, method=method, policy=policy,
                                   **metrics(reference, image)))
    write_csv(output / 'metrics.csv', scores)

    subprocess.run([sys.executable, str(Path(__file__).with_name('run_imagej.py')),
                    '--run', str(output), '--preserve-psf-mass'], check=True)
    figure, axes = plt.subplots(2, 5, figsize=(11, 4.15), layout='constrained')
    columns = {'reference': 'Reference', 'observed': 'Observed', 'TRIF': 'TRIF',
               'RL': 'RL', 'RLTV': 'RLTV'}
    for row, noise in enumerate(SEEDS):
        folder = output / 'test' / f'epflbars_{noise}'
        for ax, (method, label) in zip(axes[row], columns.items()):
            ax.imshow(tiff(folder / f'{method}.tif'), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(label, fontsize=12)
        axes[row, 0].text(-.12, .5, noise.capitalize(), ha='center', va='center',
                          rotation='vertical', transform=axes[row, 0].transAxes)
    figure.savefig(output / 'projection.png', dpi=190)
    plt.close(figure)
    print(f'PASS: EPFL Hollow Bars, projection error={projection_error:.2e}; {output}')


if __name__ == '__main__':
    main()
