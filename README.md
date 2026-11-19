# Image Restoration by Deconvolution

Digital Image Processing course project, class E23TTNT02, Posts and Telecommunications Institute of Technology. The project implements five non-blind 2D solvers in Python and checks them against DeconvolutionLab2 2.1.2 running in ImageJ 1.54p.

- [Group 07 course report](<report/NHÓM 07 - BÁO CÁO BÀI TẬP LỚN.pdf>) and [standalone LaTeX sources](report/bao-cao-latex.zip)
- [Reproduction instructions](code/README.md) and [report build instructions](report/README.md)
- [Experiment evidence](evidence/README.md), [Python test metrics](evidence/test_metrics.csv), and [independent ImageJ metrics](evidence/imagej/imagej_metrics.csv)
- [Technical and academic review](docs/05_ra_soat_hoc_thuat.md), with the original team drafts in [docs](docs/README.md)

## Results at a glance

On ten BSDS500 test images under each noise model, Python RLTV achieved **25.44 dB with Gaussian noise** and **24.89 dB with Poisson noise**, gains of 1.76 and 1.92 dB over the observations. Four like-for-like solver comparisons matched ImageJ within `2.3e-5` raw intensity units. The transferred ranking changed on the independent EPFL Hollow Bars projection, where RIF achieved the highest PSNR for both noise cases. The paper reports the different RIF, RL, and RLTV implementation conventions explicitly.

## Project root

~~~text
BTL/
├── code/
│   ├── deconv/        # Python forward model, solvers, and metrics
│   ├── imagej/        # Java launcher and independent metrics
│   ├── scripts/       # Preparation, experiments, and audits
│   └── tests/         # Numerical and regression tests
├── report/           # Paper source, tables, figures, and PDF
├── evidence/         # Tracked numerical snapshot and manifests
├── docs/             # Technical review and original drafts
├── assets/           # Graphics referenced by original drafts
├── data/             # Local dataset downloads
└── outputs/          # Generated TIFFs, logs, figures, and CSVs
~~~

Run the commands from this root. `data/` and `outputs/` are ignored by Git and can be recreated. The report includes the main implementation listings, the complete parameter sweep, and the per-image test atlas.
