"""Reproduce the DeconvolutionLab2 run inside the ImageJ 1.x batch runtime."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

import numpy as np
from PIL import Image
from scipy import fft

from run_experiments import ROOT, METHODS, write_csv, write_json
from deconv.core import convolve, kernel_grid, save_tiff
from deconv.solvers import solve
from deconv.metrics import metrics


JARS = {
    'ij.jar': 'https://sites.imagej.net/Fiji/jars/ij-1.54p.jar-20250219110715',
    'DeconvolutionLab_2.jar':
        'https://bigwww.epfl.ch/deconvolution/deconvolutionlab2/DeconvolutionLab_2.jar',
}
JAR_SHA256 = {
    'ij.jar': '2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20',
    'DeconvolutionLab_2.jar': '37ade1a14b7063b5a1c9b53d685379cc45b3704d023db1d10dbeae92b0a6762c',
}


def tiff(path):
    with Image.open(path) as image:
        if image.mode != 'F':
            raise ValueError(f'Expected float TIFF: {path}')
        return np.asarray(image, dtype=float)


def latest_run():
    return ROOT / 'outputs' / (ROOT / 'outputs/LATEST.txt').read_text().strip()


def local_jar(path, url):
    expected = JAR_SHA256[path.name]
    if not path.is_file():
        temporary = path.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=60) as response:
            temporary.write_bytes(response.read())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Downloaded JAR checksum mismatch: {url}')
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f'Cached JAR checksum mismatch: {path}')
    return dict(url=url, sha256=digest)


def match_rif(g, h, lam, eps):
    """DL2 2.1.2 uses a spectral Laplacian for these even-size inputs."""
    spectrum = fft.rfft2(kernel_grid(h, g.shape))
    wx = 2 * np.pi * fft.rfftfreq(g.shape[1])
    wy = 2 * np.pi * fft.fftfreq(g.shape[0])[:, None]
    denominator = spectrum**2 + lam * (wx**2 + wy**2)**2
    transfer = spectrum * denominator.conj() / np.maximum(eps, abs(denominator)**2)
    return fft.irfft2(fft.rfft2(g) * transfer, s=g.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--preserve-psf-mass', action='store_true',
                        help='Skip Java renormalization of a float TIFF already normalized to unit mass')
    args = parser.parse_args()
    run = args.run.resolve() if args.run else latest_run()
    config = json.loads((run / 'config_frozen_before_test.json').read_text())
    report = run / 'imagej'
    report.mkdir(exist_ok=False)
    toolchain = ROOT / 'outputs/toolchain'
    toolchain.mkdir(parents=True, exist_ok=True)
    versions = {name: local_jar(toolchain / name, url) for name, url in JARS.items()}
    write_json(report / 'jars.json', versions)
    classes = report / 'classes'
    classes.mkdir()
    classpath = os.pathsep.join(str(p) for p in [classes, *(toolchain / name for name in JARS)])

    cases = {}
    jobs = []
    norm = 'no' if args.preserve_psf_mass else '1'
    for folder in sorted((run / 'test').iterdir()):
        image_id, noise = folder.name.split('_')
        ref, g, h = (tiff(folder / f'{name}.tif') for name in ('reference', 'observed', 'psf'))
        h /= h.sum()
        dest = report / folder.name
        save_tiff(dest / 'psf_centered.tif', fft.fftshift(kernel_grid(h, g.shape)))
        cases[folder.name] = (folder, ref, g, h)

        for method in ['observed', *METHODS]:
            params = config['selected'][noise].get(method, {})
            parameters = [params[key] for key in ('iterations', 'lam') if key in params]
            command = ' '.join(map(str, [method, *parameters]))
            eps = config['nif_floor'] if method == 'NIF' else 1e-6
            jobs.append((folder.name, method, folder / 'reference.tif', folder / 'observed.tif',
                         dest / 'psf_centered.tif', dest, command, eps, norm))

    # Asymmetric impulse at an edge exposes phase, orientation and circular wrapping.
    probe = report / 'operator_probe'
    impulse = np.zeros((32, 32))
    impulse[0, 31] = 1
    asymmetric = np.array([[0., .1, .2], [.3, .4, 0.], [0., 0., 0.]])
    expected = convolve(impulse, asymmetric)
    save_tiff(probe / 'reference.tif', expected)
    save_tiff(probe / 'observed.tif', impulse)
    save_tiff(probe / 'psf_centered.tif', fft.fftshift(kernel_grid(asymmetric, impulse.shape)))
    jobs.append(('operator_probe', 'CONV', probe / 'reference.tif', probe / 'observed.tif',
                 probe / 'psf_centered.tif', probe, 'CONV', 1e-6, norm))

    job_file = report / 'jobs.tsv'
    with job_file.open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream, delimiter='\t', lineterminator='\n').writerows(jobs)
    subprocess.run(['javac', '-encoding', 'UTF-8', '-cp', classpath, '-d', str(classes),
                    str(ROOT / 'code/imagej/BatchReference.java')], check=True)
    with (report / 'plugin.log').open('w', encoding='utf-8') as log:
        subprocess.run(['java', '-Xmx1024m', '-Djava.awt.headless=true', '-cp', classpath,
                        'BatchReference', str(job_file)], stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    if re.search(r'(?m)^Exception|^\s+at [a-zA-Z].+\.java:\d+', (report / 'plugin.log').read_text()):
        raise RuntimeError('Plugin reported an exception; inspect plugin.log')

    with (report / 'imagej_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected_keys = {(case, method, policy) for case, method, *_ in jobs for policy in ('raw', 'clip01')}
    keys = {(r['case'], r['method'], r['policy']) for r in rows}
    if len(rows) != len(expected_keys) or keys != expected_keys:
        raise ValueError('Incomplete or duplicate ImageJ metric records')

    metric_errors = {key: 0. for key in ('mse', 'psnr', 'ssim', 'dl2_psnr_raw')}
    for row in rows:
        case, method, policy = (row[key] for key in ('case', 'method', 'policy'))
        if case == 'operator_probe':
            ref, result = tiff(probe / 'reference.tif'), tiff(probe / 'CONV.tif')
        else:
            folder, ref, g, _ = cases[case]
            result = g if method == 'observed' else tiff(report / case / f'{method}.tif')
        for key, value in metrics(ref, result if policy == 'raw' else np.clip(result, 0, 1)).items():
            recorded = float(row[key])
            error = 0.0 if value == recorded else abs(value - recorded)
            if not np.isfinite(error):
                raise ValueError(f'Invalid metric in {case}/{method}/{policy}: {key}')
            metric_errors[key] = max(metric_errors[key], error)
        if policy == 'raw':
            plugin_score = float(row['dl2_psnr_raw'])
            expected_score = float(row['psnr']) + 20 * np.log10(ref.max())
            error = 0.0 if expected_score == plugin_score else abs(expected_score - plugin_score)
            if not np.isfinite(error):
                raise ValueError(f'Invalid native PSNR in {case}/{method}')
            metric_errors['dl2_psnr_raw'] = max(metric_errors['dl2_psnr_raw'], error)
    if max(metric_errors.values()) >= 1e-7:
        raise ValueError(f'Metric disagreement: {metric_errors}')

    parity = []
    for case, (folder, _, g, h) in cases.items():
        _, noise = case.split('_')
        for method in METHODS:
            plugin = tiff(report / case / f'{method}.tif')
            normal = tiff(folder / f'{method}.tif')
            params = config['selected'][noise][method]
            same = {'NIF': lambda: normal, 'TRIF': lambda: normal,
                    'RIF': lambda: match_rif(g, h, params['lam'], 1e-6),
                    'RL': lambda: solve(g, h, 'RL', **params, eps=1e-6, imagej_rl=True)}
            equivalent = same[method]() if method in same else None
            difference = plugin - normal
            parity.append(dict(case=case, method=method,
                               default_rmse=float(np.sqrt(np.mean(difference**2))),
                               default_max=float(abs(difference).max()),
                               matched_max=None if equivalent is None else float(abs(plugin-equivalent).max())))
    parity_limits = {'NIF': 1e-4, 'TRIF': 1e-5, 'RIF': 1e-5, 'RL': 5e-5}
    parity_max = {method: max(r['matched_max'] for r in parity if r['method'] == method)
                  for method in parity_limits}
    probe_error = float(abs(expected - tiff(probe / 'CONV.tif')).max())
    if probe_error >= 1e-6 or any(parity_max[m] >= lim for m, lim in parity_limits.items()):
        raise ValueError(f'Plugin agreement outside tolerances: CONV={probe_error}; {parity_max}')

    write_csv(report / 'parity.csv', parity)
    write_json(report / 'audit.json', dict(status='PASS', cases=len(cases), outputs=len(cases) * len(METHODS),
               metric_rows=len(rows), metric_max_absolute_error=metric_errors, convolution_max=probe_error,
               matched_max_error=parity_max, tolerances=parity_limits, plugin_psf_normalization=norm))
    print(f'PASS: {len(cases)} cases, {len(rows)} metric rows and impulse-PSF test. {report}')


if __name__ == '__main__':
    main()
