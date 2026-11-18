# TV2 — BSDS500 deconvolution

Run from the repository root with the existing `.venv`; no added dependencies.

```powershell
.\.venv\Scripts\python.exe team-project/code/scripts/prepare_data.py
.\.venv\Scripts\python.exe team-project/code/scripts/check.py
.\.venv\Scripts\python.exe team-project/code/scripts/run_experiments.py
```

`prepare_data.py` downloads the official Berkeley archive, selects five validation and ten test JPEGs in lexical order, and records SHA256. It extracts only regular JPEG members by basename, never archive paths. Failed downloads stay `.part` and curl can resume them.

`run_experiments.py` creates a new timestamped run. Validation selects parameters; the configuration is saved before test evaluation. No fitting on test. These are synthetic restoration experiments using BSDS500 reference photographs, not the official segmentation benchmark.

## Layout

```text
team-project/
  code/deconv/           # core.py, solvers.py, metrics.py
  code/tests/            # independent numerical checks and hand examples
  code/scripts/          # reproducible entry points
  data/bsds500/          # official archive, 15 JPEGs, manifest
  outputs/
    verification/       # actual unittest transcripts
    LATEST.txt          # latest completed experiment folder
    bsds500_<timestamp>/
      config_frozen_before_test.json
      environment.json
      manifest.json
      validation_sweep.csv
      test_metrics.csv
      test_summary.csv
      benchmark.csv
      gallery.html
      val/<id>_<noise>/  # reference, blurred, observed, PSF and input.json
      test/<id>_<noise>/ # same inputs plus five raw TIFF results, PNG, comparison, ROI
```

TIFF is float32 uncompressed and retains out-of-range results. Computation is float64. PNG and displayed comparisons clip to [0,1]; PSF PNG is scaled only for viewing. Use `psf.tif` for computation. Test CSV includes both raw and clipped policies with data_range=1. SSIM uses Gaussian 11×11 sigma 1.5, population covariance, valid-window mean.

Optimization compares full complex FFT/repeated OTF against real FFT/cached OTF using identical math. Timing excludes plotting, metrics and file I/O; median of seven alternating-order repetitions after warmup. Tracemalloc is measured separately and is not process RSS.

## Fiji handoff

Open the same reference.tif, observed.tif and psf.tif from a test case, using 32-bit intensity unchanged. Match periodic/no-extra-padding convention, no apodization, PSF normalization=1 and save raw float output. Record all settings from the actual plugin version, including epsilon and initialization. Algorithm name alone does not establish equivalence: NIF floor, RIF Laplacian and RLTV discretization may differ. Run through Fiji/ImageJ and retain screenshots/logs before claiming platform validation.

Sources: [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), [Sage et al. 2017](https://bigwww.epfl.ch/deconvolution/deconvolutionlab2/sage1701p.pdf).
