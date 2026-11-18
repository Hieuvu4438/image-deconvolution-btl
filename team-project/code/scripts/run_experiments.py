"""Validation-only selection, frozen test evaluation, benchmark and HTML gallery."""
from pathlib import Path
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import platform
import sys
import time
import tracemalloc
import statistics
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from deconv.core import read_reference, gaussian_psf, degrade, save_tiff, save_png
from deconv.solvers import solve
from deconv.metrics import metrics

METHODS=['NIF','TRIF','RIF','RL','RLTV']
CANDIDATES={'NIF':[{}], 'TRIF':[{'lam':x} for x in [.0001,.001,.01,.02]],
            'RIF':[{'lam':x} for x in [.0001,.001,.005,.02]],
            'RL':[{'iterations':x} for x in [5,10,20,40]],
            'RLTV':[{'iterations':20,'lam':x,'delta':.01} for x in [.0001,.001]]}

def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main() -> None:
    stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'outputs'/f'bsds500_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'data/bsds500/manifest.json').read_text(encoding='utf-8'))
    write_json(out/'manifest.json',manifest)
    write_json(out/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                                         platform=platform.platform(),processor=platform.processor(),
                                         timestamp=stamp,timezone='Asia/Ho_Chi_Minh'))
    h=gaussian_psf()
    samples={}
    for item in manifest['images']:
        path=ROOT/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        ref=read_reference(path)
        for noise in ['gaussian','poisson']:
            seed=20261118+int(item['id'])+(1000000 if noise=='poisson' else 0)
            blurred,g=degrade(ref,h,noise,seed)
            folder=out/item['split']/f"{item['id']}_{noise}"
            folder.mkdir(parents=True)
            for name,array in [('reference',ref),('blurred',blurred),('observed',g),('psf',h)]:
                save_tiff(folder/f'{name}.tif',array)
                save_png(folder/f'{name}.png',array if name!='psf' else array/array.max())
            write_json(folder/'input.json',dict(id=item['id'],split=item['split'],seed=seed,noise=noise,
                                               sigma_noise=3/255 if noise=='gaussian' else None,
                                               photon_peak=700 if noise=='poisson' else None,
                                               preprocessing='Pillow L, center crop 256, divide 255',
                                               psf_size=17,psf_sigma=1.9,boundary='periodic',
                                               negative_gaussian_clipped=True if noise=='gaussian' else False))
            samples[(item['split'],item['id'],noise)]=(ref,g,folder)
    print('Inputs exported; selecting parameters on validation only.',flush=True)
    sweep=[];selected={}
    for noise in ['gaussian','poisson']:
        selected[noise]={}
        for method in METHODS:
            choices=[]
            for params in CANDIDATES[method]:
                scores=[]
                for (split,image_id,kind),(ref,g,_) in samples.items():
                    if split!='val' or kind!=noise: continue
                    result=solve(g,h,method,**params)
                    score=metrics(ref,np.clip(result,0,1))
                    scores.append(score['psnr'])
                    sweep.append(dict(id=image_id,noise=noise,method=method,params=json.dumps(params),**score))
                choices.append((float(np.mean(scores)),params))
            selected[noise][method]=max(choices,key=lambda x:x[0])[1]
    write_csv(out/'validation_sweep.csv',sweep)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for row,noise in enumerate(['gaussian','poisson']):
        for col,method in enumerate(['TRIF','RIF','RL','RLTV']):
            values=[]
            for params in CANDIDATES[method]:
                scores=[r['psnr'] for r in sweep if r['noise']==noise and r['method']==method and r['params']==json.dumps(params)]
                values.append((params.get('lam',params.get('iterations')),float(np.mean(scores))))
            ax=axes[row,col];ax.plot(*zip(*values),marker='o')
            if method!='RL': ax.set_xscale('log')
            ax.set_title(f'{noise} / {method}');ax.set_xlabel('K' if method=='RL' else 'lambda')
            ax.set_ylabel('Validation mean PSNR')
    fig.tight_layout();fig.savefig(out/'validation_sweep.png',dpi=150);plt.close(fig)
    config=dict(selected=selected,selection='mean clipped PSNR on five validation images only',
                psf_size=17,psf_sigma=1.9,boundary='periodic',dtype='float64',nif_floor=.0001,
                rl_epsilon=1e-12,rl_init='observed mean',rltv_delta=.01,
                ssim='11x11 Gaussian sigma 1.5; population covariance; valid window centers; range=1',
                metrics='raw and clipped [0,1]; reference grayscale JPEG crop; no linear-light claim',
                candidates=CANDIDATES)
    write_json(out/'config_frozen_before_test.json',config)
    print('Configuration frozen; evaluating test subset.',flush=True)
    rows=[];gallery=[]
    for (split,image_id,noise),(ref,g,folder) in samples.items():
        if split!='test': continue
        images={'reference':ref,'observed':g}
        for method in ['observed']+METHODS:
            start=time.perf_counter()
            result=g if method=='observed' else solve(g,h,method,**selected[noise][method])
            seconds=time.perf_counter()-start
            images[method]=result
            if method!='observed':
                save_tiff(folder/f'{method}.tif',result);save_png(folder/f'{method}.png',result)
            for policy in ['raw','clip01']:
                score=metrics(ref,result if policy=='raw' else np.clip(result,0,1))
                rows.append(dict(id=image_id,noise=noise,method=method,policy=policy,
                                 seconds=seconds,minimum=float(result.min()),maximum=float(result.max()),**score))
        fig,axes=plt.subplots(2,4,figsize=(14,7))
        for ax,(name,array) in zip(axes.flat,images.items()):
            ax.imshow(np.clip(array,0,1),cmap='gray',vmin=0,vmax=1);ax.set_title(name);ax.axis('off')
        axes.flat[-1].axis('off');fig.suptitle(f'BSDS500 {image_id} / {noise} / shared display [0,1]')
        fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
        # Fixed central ROI, selected independently of method performance.
        fig,axes=plt.subplots(1,7,figsize=(16,3))
        for ax,(name,array) in zip(axes,images.items()):
            ax.imshow(np.clip(array[96:160,96:160],0,1),cmap='gray',vmin=0,vmax=1)
            ax.set_title(name);ax.axis('off')
        fig.tight_layout();fig.savefig(folder/'roi.png',dpi=140);plt.close(fig)
        relative=folder.relative_to(out).as_posix()
        gallery.append(f'<h2>{image_id} / {noise}</h2><img src="{relative}/comparison.png"><img src="{relative}/roi.png">')
    write_csv(out/'test_metrics.csv',rows)
    summary=[]
    for noise in ['gaussian','poisson']:
        for method in ['observed']+METHODS:
            r=[x for x in rows if x['noise']==noise and x['method']==method and x['policy']=='clip01']
            summary.append(dict(noise=noise,method=method,n=len(r),
                                psnr_mean=float(np.mean([x['psnr'] for x in r])),
                                psnr_std=float(np.std([x['psnr'] for x in r],ddof=1)),
                                ssim_mean=float(np.mean([x['ssim'] for x in r]))))
    write_csv(out/'test_summary.csv',summary)
    print('Test outputs saved; benchmarking baseline versus optimized.',flush=True)
    _,g,_=next(value for key,value in samples.items() if key[0]=='val' and key[2]=='poisson')
    bench=[]
    for method in METHODS:
        params=selected['poisson'][method]
        base=solve(g,h,method,optimized=False,**params);opt=solve(g,h,method,**params)
        np.testing.assert_allclose(base,opt,atol=1e-9,rtol=1e-9)
        times={False:[],True:[]};peaks={}
        for repeat in range(7):
            for optimized in ([False,True] if repeat%2==0 else [True,False]):
                start=time.perf_counter();solve(g,h,method,optimized=optimized,**params)
                times[optimized].append(time.perf_counter()-start)
        for optimized in [False,True]:
            tracemalloc.start();solve(g,h,method,optimized=optimized,**params)
            peaks[optimized]=tracemalloc.get_traced_memory()[1]/2**20;tracemalloc.stop()
        a,b=statistics.median(times[False]),statistics.median(times[True])
        bench.append(dict(method=method,params=json.dumps(params),baseline_s=a,optimized_s=b,speedup=a/b,
                          baseline_traced_mib=peaks[False],optimized_traced_mib=peaks[True],
                          max_abs_error=float(abs(base-opt).max()),repeats=7,
                          baseline_times=json.dumps(times[False]),optimized_times=json.dumps(times[True])))
    write_csv(out/'benchmark.csv',bench)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,noise in zip(axes,['gaussian','poisson']):
        r=[x for x in summary if x['noise']==noise]
        ax.bar([x['method'] for x in r],[x['psnr_mean'] for x in r],yerr=[x['psnr_std'] for x in r],capsize=3)
        ax.set_title(f'{noise}: 10 test images, mean +/- SD');ax.set_ylabel('PSNR (dB), clipped')
        ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(out/'quality_summary.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(5)
    ax.bar(x-.18,[r['baseline_s']*1000 for r in bench],.36,label='Full FFT + repeated OTF')
    ax.bar(x+.18,[r['optimized_s']*1000 for r in bench],.36,label='Real FFT + cached OTF')
    ax.set_xticks(x,METHODS);ax.set_ylabel('Median milliseconds (7 runs)');ax.legend()
    fig.tight_layout();fig.savefig(out/'benchmark.png',dpi=160);plt.close(fig)
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>BSDS500 TV2</title>'
        '<style>body{max-width:1500px;margin:auto;font-family:Arial}img{width:100%}</style>'
        '<h1>BSDS500 — Python deconvolution</h1><p>10 fixed test images; synthetic degradation. '
        'Raw TIFF is preserved; PNG display clips to [0,1]. Fiji comparison pending.</p>'
        '<img src="validation_sweep.png"><img src="quality_summary.png"><img src="benchmark.png">'+''.join(gallery),encoding='utf-8')
    (ROOT/'outputs/LATEST.txt').write_text(out.name,encoding='utf-8')
    print(f'COMPLETE {out.name}',flush=True)

if __name__=='__main__': main()
