# Experiment evidence

`RUN.txt` identifies the audited run used in the PDF. The files here are copies of the numerical data and manifests for that run. Raw TIFFs and full runtime logs are saved under the local `outputs/` run directory.

| Location | Contents |
|---|---|
| Root CSV and JSON files | BSDS500 manifest, frozen settings, validation results, main test metrics, Python audit, and benchmark |
| `imagej/` | Independently scored ImageJ TIFF outputs, native PSNR, raw parity and the JAR versions |
| `sensitivity/` | PSF and boundary mismatch, full/inner regions, and phantom scores |
| `epfl-bars/` | Public archive checksums, projection protocol, and Python/ImageJ results for the optical PSF |

The project scripts regenerate the full run, including the original TIFFs and logs, from the official data sources. [Workflow](../code/README.md)
