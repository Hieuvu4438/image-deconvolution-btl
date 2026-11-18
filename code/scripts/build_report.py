"""Build standalone LaTeX assets from the verified experiment directory."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import fft

from run_experiments import ROOT, METHODS, plt
from run_imagej import latest_run, tiff
from run_sensitivity import SCENARIOS
from deconv.core import gaussian_psf, horizontal_motion_psf


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def num(value, digits=2):
    return f'{float(value):.{digits}f}'


def sci(value):
    mantissa, exponent = f'{float(value):.2e}'.split('e')
    return '$' + fr'{mantissa}\times 10^{{{int(exponent)}}}' + '$'


def table(path, rows):
    body = '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    path.write_text(body + '\n' + r'\bottomrule' + '\n', encoding='utf-8')


def format_params(params):
    parts = []
    if 'iterations' in params:
        parts.append(f"K={params['iterations']}")
    if 'lam' in params:
        parts.append(fr"\lambda={params['lam']:g}")
    return '$' + r',\quad '.join(parts or [r'\tau=10^{-4}']) + '$'


def plot_figures(run, figures, values, image_id, config):
    plt.rcParams.update({'font.size': 11, 'font.family': 'STIXGeneral',
                         'axes.labelcolor': 'black', 'axes.titlecolor': 'black'})
    palette = ['#9b3c3c', '#395f8c', '#a16a28', '#74406d', '#1e7867']
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), layout='constrained', sharey=True)
    for ax, noise in zip(axes, ['gaussian', 'poisson']):
        for i, (method, color) in enumerate(zip(METHODS, palette)):
            scores = values[noise, method]
            ax.scatter(scores, np.full(scores.size, i) + np.linspace(-.17, .17, scores.size),
                       color=color, alpha=.48, s=15, linewidths=0)
            ax.scatter(scores.mean(), i, color=color, edgecolor='white', s=60, marker='D', zorder=3)
        ax.axvline(0, lw=1, color='.45')
        ax.set(title=noise.capitalize(), xlabel='Restored PSNR minus observed PSNR (dB)',
               yticks=range(5), yticklabels=METHODS)
        ax.grid(axis='x', alpha=.25)
    axes[0].invert_yaxis()
    fig.savefig(figures/'paired.png', dpi=190)
    plt.close(fig)

    # Centering only changes phase. The figure compares OTF magnitudes.
    h, motion = gaussian_psf(), horizontal_motion_psf()
    amplitude = abs(fft.rfft(h.sum(axis=0), n=512))
    freq = fft.rfftfreq(512)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), layout='constrained')
    axes[0].plot(freq, amplitude, label='Gaussian', color=palette[1])
    axes[0].plot(freq, abs(fft.rfft(motion.ravel(), n=512)), label='9-pixel motion', color=palette[-1])
    axes[0].set(xlabel='Horizontal frequency (cycles/pixel)', ylabel='OTF magnitude', ylim=(0, 1.03))
    axes[1].semilogy(freq, 1 / np.maximum(amplitude, 1e-20), ':', color='.4', label='Algebraic inverse')
    axes[1].semilogy(freq, amplitude / np.maximum(amplitude**2, 1e-4), color=palette[0], label='NIF')
    axes[1].semilogy(freq, amplitude / (amplitude**2 + .01), color=palette[1], label='TRIF')
    axes[1].set(xlabel='Horizontal frequency (cycles/pixel)', ylabel='Gain for the Gaussian PSF', ylim=(1e-4, 1e6))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(figures/'frequency.png', dpi=190)
    plt.close(fig)

    sweep = read_csv(run / 'validation_sweep.csv')
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.1), layout='constrained')
    for row, noise in enumerate(['gaussian', 'poisson']):
        for col, method in enumerate(METHODS[1:]):
            ax = axes[row, col]
            settings = config['candidates'][method]
            if method == 'RLTV':
                chosen_k = config['selected'][noise][method]['iterations']
                settings = [p for p in settings if p['iterations'] == chosen_k]
            field = 'iterations' if method == 'RL' else 'lam'
            means = []
            for params in settings:
                scores = [float(r['psnr']) for r in sweep if r['noise'] == noise
                          and r['method'] == method and json.loads(r['params']) == params]
                means.append(np.mean(scores))
            xs = [p[field] for p in settings]
            ax.plot(xs, means, 'o-', color=palette[col+1], lw=1.4, markersize=4)
            title = f'{method} (K={chosen_k})' if method == 'RLTV' else method
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('K' if field == 'iterations' else r'$\lambda$', fontsize=11)
            if field == 'lam':
                ax.set_xscale('log')
            if col == 0:
                ax.set_ylabel(f'{noise.capitalize()}\nPSNR (dB)', fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(alpha=.22)
    fig.savefig(figures/'validation.png', dpi=190)
    plt.close(fig)

    case = image_id + '_poisson'
    folders = [run / 'test' / case, run / 'imagej' / case]
    methods = ['reference', 'observed', *METHODS]
    labels = ['Reference', 'Observed', *METHODS]
    fig, axes = plt.subplots(2, 7, figsize=(11.6, 3.6), layout='constrained')
    for index, axes_row in enumerate(axes):
        for col, (ax, method) in enumerate(zip(axes_row, methods)):
            source = folders[0] if col < 2 else folders[index]
            image = tiff(source / f'{method}.tif')
            ax.imshow(image[96:160, 96:160], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if index == 0:
                ax.set_title(labels[col])
        axes_row[0].text(-.13, .5, ['Python', 'ImageJ'][index], transform=axes_row[0].transAxes,
                         rotation='vertical', ha='center', va='center')
    fig.savefig(figures/'platform-roi.png', dpi=190)
    plt.close(fig)


def plot_rl_fit(run, tables, figures):
    rows = read_csv(run / 'validation_rl_fit.csv')
    iterations = sorted({int(row['iterations']) for row in rows})
    measures = ['psnr', 'kl_mean']
    stats = [[np.mean([float(row[key]) for row in rows if int(row['iterations']) == k])
              for key in measures] for k in iterations]
    if len(rows) != 25:
        raise ValueError('Expected five RL settings for five Poisson validation images')
    table(tables / 'rl-fit.tex', [[k, num(psnr, 3), sci(kl)] for k, (psnr, kl) in zip(iterations, stats)])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.0), layout='constrained')
    for col, (ax, label, color) in enumerate(zip(
            axes, ['PSNR against clean reference (dB)', 'Observation KL per pixel'],
            ['#395f8c', '#1e7867'])):
        ax.plot(iterations, [row[col] for row in stats], 'o-', color=color)
        ax.set(xlabel='Richardson--Lucy iterations', ylabel=label, xticks=iterations)
        ax.grid(alpha=.22)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.savefig(figures/'rl-fit.png', dpi=185)
    plt.close(fig)


def build_atlas(run, tables, figures, primary, imagej, image_ids, methods):
    py = {(r['id'], r['noise'], r['method']): r for r in primary if r['policy'] == 'clip01'}
    ij = {(r['case'], r['method']): r for r in imagej if r['policy'] == 'clip01'}
    sections = []
    for image_id in image_ids:
        fig, axes = plt.subplots(4, 6, figsize=(11, 7.65), layout='constrained')
        rows = []
        for index, noise in enumerate(['gaussian', 'poisson']):
            case = f'{image_id}_{noise}'
            test, plugin = run / 'test' / case, run / 'imagej' / case
            entries = [(test, 'reference', 'Reference'), (test, 'observed', 'Observed'),
                       (test, 'TRIF', 'Python TRIF'), (test, 'RLTV', 'Python RLTV'),
                       (plugin, 'RIF', 'ImageJ RIF'), (plugin, 'RLTV', 'ImageJ RLTV')]
            for col, (folder, method, label) in enumerate(entries):
                a = tiff(folder / f'{method}.tif')
                for row, data in [(index, a), (index + 2, a[96:160, 96:160])]:
                    ax = axes[row, col]
                    ax.imshow(data, cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                    if row == 0:
                        ax.set_title(label, fontsize=12)
            axes[index, 0].text(-.10, .5, noise.capitalize() + ' / full',
                                transform=axes[index, 0].transAxes, rotation='vertical',
                                ha='center', va='center')
            axes[index+2, 0].text(-.10, .5, noise.capitalize() + ' / ROI',
                                  transform=axes[index+2, 0].transAxes, rotation='vertical',
                                  ha='center', va='center')
            baseline = float(py[(image_id, noise, 'observed')]['psnr'])
            for method, label in methods.items():
                p, j = py[(image_id, noise, method)], ij[(case, method)]
                pp, jp = float(p['psnr']), float(j['psnr'])
                rows.append([noise.capitalize() if method == 'observed' else '', label,
                             num(pp), num(p['ssim'], 4), num(pp-baseline),
                             num(jp), num(j['ssim'], 4), num(jp-baseline)])
        fig.savefig(figures / f'atlas-{image_id}.png', dpi=155)
        plt.close(fig)
        table(tables / f'atlas-{image_id}.tex', rows)
        page_break = r'\clearpage' if sections else ''
        image_width = r'\linewidth' if sections else r'.92\linewidth'
        sections.append(fr"""
{page_break}
\subsection{{BSDS500 image {image_id}}}
\begin{{center}}
\includegraphics[width={image_width}]{{atlas-{image_id}.png}}
\captionof{{figure}}{{Full-frame and central-ROI comparison for image {image_id}.}}
\end{{center}}
\begin{{center}}
\small\setlength{{\tabcolsep}}{{3.3pt}}
\captionof{{table}}{{Clipped whole-image results. Gain is relative to the observation (dB).}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
& & \multicolumn{{3}}{{c}}{{Python}} & \multicolumn{{3}}{{c}}{{DeconvolutionLab2}}\\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}
Noise & Method & PSNR & SSIM & Gain & PSNR & SSIM & Gain\\\midrule
\input{{tables/atlas-{image_id}}}
\end{{tabular}}
\end{{center}}
""")
    (tables / 'atlas.tex').write_text('\n'.join(sections), encoding='utf-8')


def snapshot_evidence(run):
    files = [
        'manifest.json', 'environment.json', 'config_frozen_before_test.json', 'artifact_audit.json',
        'validation_sweep.csv', 'validation_rl_fit.csv', 'test_metrics.csv', 'test_summary.csv',
        'benchmark.csv', 'imagej/audit.json', 'imagej/jars.json', 'imagej/imagej_metrics.csv',
        'imagej/parity.csv', 'sensitivity/protocol.json', 'sensitivity/metrics.csv',
        'sensitivity/summary.csv', 'sensitivity/phantom_metrics.csv', 'epfl-bars/manifest.json',
        'epfl-bars/metrics.csv', 'epfl-bars/imagej/imagej_metrics.csv',
        'epfl-bars/imagej/audit.json', 'epfl-bars/imagej/parity.csv',
    ]
    dest = ROOT / 'evidence'
    for name in files:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, path)
    (dest / 'RUN.txt').write_text(run.name + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve() if args.run else latest_run()
    tables, figures = ROOT / 'report/tables', ROOT / 'report/figures'
    for path in (tables, figures):
        path.mkdir(parents=True, exist_ok=True)
    py_audit, ij_audit = [json.loads((run / path).read_text())
                          for path in ('artifact_audit.json', 'imagej/audit.json')]
    epfl = run / 'epfl-bars'
    epfl_audit = json.loads((epfl / 'imagej/audit.json').read_text())
    epfl_manifest = json.loads((epfl / 'manifest.json').read_text())
    if any(audit['status'] != 'PASS' for audit in (py_audit, ij_audit, epfl_audit)):
        raise ValueError('All verification stages must pass before report generation')
    config = json.loads((run / 'config_frozen_before_test.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    image_ids = [r['id'] for r in manifest['images'] if r['split'] == 'test']
    environment = json.loads((run / 'environment.json').read_text())
    benchmark = read_csv(run / 'benchmark.csv')
    speedups = [float(row['speedup']) for row in benchmark]
    definitions = {
        'RunID': run.name.replace('_', r'\_'),
        'PythonVersion': environment['python'].split()[0],
        'NumPyVersion': environment['numpy'], 'SciPyVersion': environment['scipy'],
        'JVMVersion': (run / 'imagej/plugin.log').read_text().splitlines()[0].split('Java ')[-1],
        'TestIDs': ', '.join(image_ids),
        'ValidationIDs': ', '.join(r['id'] for r in manifest['images'] if r['split'] == 'val'),
        'ProbeError': sci(ij_audit['convolution_max']),
        'PythonMetricError': sci(py_audit['max_saved_float32_metric_difference']),
        'MetricPSNRGap': sci(ij_audit['metric_max_absolute_error']['psnr']),
        'MetricSSIMGap': sci(ij_audit['metric_max_absolute_error']['ssim']),
        'MetricMSEGap': sci(ij_audit['metric_max_absolute_error']['mse']),
        'NativePSNRGap': sci(ij_audit['metric_max_absolute_error']['dl2_psnr_raw']),
        'BenchmarkSpeedRange': num(min(speedups)) + '--' + num(max(speedups)),
        'BenchmarkMaxError': sci(max(float(row['max_abs_error']) for row in benchmark)),
        'EpflProjectionError': sci(epfl_manifest['projection_max_abs_error']),
        'EpflMassPercent': num(100 * epfl_manifest['psf_mass_17']),
        'EpflParityError': sci(max(epfl_audit['matched_max_error'].values())),
    }
    (tables / 'run.tex').write_text('\n'.join(
        fr'\newcommand{{\{k}}}{{{v}}}' for k, v in definitions.items()) + '\n', encoding='utf-8')
    methods = {'observed': 'Observed', **dict(zip(METHODS, METHODS))}
    noises = ['gaussian', 'poisson']
    settings = [[method, *[format_params(config['selected'][noise][method]) for noise in noises]]
                for method in METHODS]
    table(tables / 'settings.tex', settings)

    primary, imagej = [read_csv(run / f) for f in ['test_metrics.csv', 'imagej/imagej_metrics.csv']]
    summary, gaps, values = [], [], {}
    for noise in noises:
        g = {r['id']: float(r['psnr']) for r in primary
             if (r['noise'], r['method'], r['policy']) == (noise, 'observed', 'clip01')}
        for method, name in methods.items():
            py = [r for r in primary if (r['noise'], r['method'], r['policy']) == (noise, method, 'clip01')]
            ij = [r for r in imagej if r['case'].endswith('_'+noise)
                  and (r['method'], r['policy']) == (method, 'clip01')]
            raw = [r for r in primary if (r['noise'], r['method'], r['policy']) == (noise, method, 'raw')]
            gain = np.array([float(r['psnr']) - g[r['id']] for r in py])
            values[noise, method] = gain
            avg = lambda rows, field: np.mean([float(r[field]) for r in rows])
            if len(py) != len(ij) or len(py) != len(image_ids):
                raise ValueError(f'Incomplete summary for {noise}/{method}')
            sd = np.std([float(r['psnr']) for r in py], ddof=1)
            summary.append([noise.capitalize() if method == 'observed' else '', name,
                            num(avg(py, 'psnr')) + r' $\pm$ ' + num(sd),
                            num(avg(py, 'ssim'), 4), num(avg(ij, 'psnr')), num(avg(ij, 'ssim'), 4)])
            if method != 'observed':
                gaps.append([noise.capitalize() if method == 'NIF' else '', name, num(avg(raw, 'psnr')),
                             num(gain.mean()), num(gain.std(ddof=1)), f'{np.count_nonzero(gain > 0)}/10'])
    table(tables / 'primary.tex', summary)
    table(tables / 'gain.tex', gaps)
    table(tables / 'benchmark.tex', [
        [r['method'], num(1000*float(r['baseline_s'])), num(1000*float(r['optimized_s'])),
         num(r['speedup']), num(r['baseline_traced_mib']), num(r['optimized_traced_mib']),
        sci(r['max_abs_error'])] for r in benchmark])
    parity = read_csv(run / 'imagej/parity.csv')
    table(tables / 'parity.tex', [
        [method, sci(max(float(r['default_max']) for r in parity if r['method'] == method)),
         sci(ij_audit['matched_max_error'][method]) if method in ij_audit['matched_max_error'] else '---',
         sci(ij_audit['tolerances'][method]) if method in ij_audit['tolerances'] else '---']
        for method in METHODS])
    sensitivity = read_csv(run / 'sensitivity/summary.csv')
    table(tables / 'sensitivity.tex', [
        [label.replace('sigma', r'\(\sigma\)'), *[num(next(r['gain_mean'] for r in sensitivity if
                          (r['scenario'], r['region'], r['method']) == (case, region, 'RLTV')))
                  for region in ('full', 'inner')]]
        for case, label in SCENARIOS.items()])
    table(tables / 'sensitivity-all.tex', [
        [label.replace('sigma', r'\(\sigma\)'), *[num(next(r['gain_mean'] for r in sensitivity if
            (r['scenario'], r['region'], r['method']) == (case, 'full', method))) for method in METHODS]]
        for case, label in SCENARIOS.items()])
    table(tables / 'phantom.tex', [
        [r['psf'].capitalize(), methods[r['method']], num(r['psnr']), num(r['ssim'], 4)]
        for r in read_csv(run / 'sensitivity/phantom_metrics.csv')])

    example_id = image_ids[0]
    example = []
    for method in ('observed', 'TRIF', 'RLTV'):
        py = next(r for r in primary if (r['id'], r['noise'], r['method'], r['policy']) ==
                  (example_id, 'poisson', method, 'clip01'))
        ij = next(r for r in imagej if (r['case'], r['method'], r['policy']) ==
                  (example_id + '_poisson', method, 'clip01'))
        scores = [num(scale * float(row[key]), digits) for row in (py, ij)
                  for key, scale, digits in [('mse', 1000, 3), ('psnr', 1, 2), ('ssim', 1, 4)]]
        example.append([methods[method], *scores])
    table(tables / 'case-example.tex', example)

    ep_py = {(r['noise'], r['method']): r for r in read_csv(epfl/'metrics.csv') if r['policy']=='clip01'}
    ep_ij = {(r['case'].split('_')[1], r['method']): r for r in read_csv(epfl/'imagej/imagej_metrics.csv')
             if r['policy']=='clip01' and r['case'].startswith('epflbars_')}
    table(tables / 'epfl.tex', [
        [noise.capitalize() if method=='observed' else '', label,
         *[num(row[key], digits) for row in [ep_py[noise, method], ep_ij[noise, method]]
           for key, digits in [('psnr', 2), ('ssim', 4)]]]
        for noise in noises for method, label in methods.items()])
    sweep = read_csv(run / 'validation_sweep.csv')
    for noise in noises:
        choices = []
        for method in METHODS:
            for params in config['candidates'][method]:
                subset = [float(r['psnr']) for r in sweep if (r['noise'], r['method']) == (noise, method)
                          and json.loads(r['params']) == params]
                if len(subset) != 5:
                    raise ValueError(f'Missing validation record: {noise}/{method}/{params}')
                chosen = params == config['selected'][noise][method]
                choices.append([method, format_params(params) + (' *' if chosen else ''), num(np.mean(subset), 3)])
        table(tables/f'sweep-{noise}.tex', choices)
    plot_figures(run, figures, values, image_ids[0], config)
    plot_rl_fit(run, tables, figures)
    build_atlas(run, tables, figures, primary, imagej, image_ids, methods)
    for source, dest in {
        'sensitivity/sensitivity.png': 'sensitivity.png', 'sensitivity/phantom.png': 'phantom.png',
        'sensitivity/profile.png': 'profile.png', 'epfl-bars/projection.png': 'epfl.png',
    }.items():
        shutil.copyfile(run / source, figures / dest)
    snapshot_evidence(run)
    print(f'Tables and figures generated from {run.name}')


if __name__ == '__main__':
    main()
