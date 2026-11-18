"""Check saved experiment coverage, raw TIFF metrics and local report assets."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from deconv.metrics import metrics

def main() -> None:
    out=ROOT/'outputs'/(ROOT/'outputs/LATEST.txt').read_text().strip()
    rows=list(csv.DictReader((out/'test_metrics.csv').open(encoding='utf-8')))
    if len(rows)!=240: raise ValueError(f'Expected 240 metric records, got {len(rows)}')
    error=0.0
    for row in rows:
        folder=out/'test'/f"{row['id']}_{row['noise']}"
        with Image.open(folder/'reference.tif') as im: ref=np.asarray(im,dtype=float)
        with Image.open(folder/f"{row['method']}.tif") as im: est=np.asarray(im,dtype=float)
        if ref.shape!=(256,256) or est.shape!=ref.shape: raise ValueError('Unexpected image shape')
        if not np.isfinite(est).all(): raise ValueError('Nonfinite saved TIFF')
        score=metrics(ref,est if row['policy']=='raw' else np.clip(est,0,1))
        for name,value in score.items():
            difference=abs(value-float(row[name]));error=max(error,difference)
            if difference>1e-4: raise ValueError(f'TIFF metric mismatch: {row} / {name}')
        for name in ['comparison.png','roi.png','psf.tif','input.json']:
            if not (folder/name).is_file(): raise FileNotFoundError(folder/name)
    result=dict(status='PASS',metric_rows=len(rows),test_cases=20,solver_outputs=100,
                max_saved_float32_metric_difference=error)
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__': main()
