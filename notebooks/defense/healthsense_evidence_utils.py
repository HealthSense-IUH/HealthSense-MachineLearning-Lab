"""Read-only evidence helpers for HealthSense defense notebooks.
Only generated charts/manifests go under outputs/healthsense_defense_evidence.
Never write to frozen experiment directories.
"""
from __future__ import annotations
import os, json, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

EVIDENCE_PREFIX = 'experiments/08_v6_locked_protocol/artifacts/'

def find_repo():
    roots=[]
    if os.getenv('HEALTHSENSE_REPO'):
        roots.append(Path(os.environ['HEALTHSENSE_REPO']).expanduser())
    cwd=Path.cwd().resolve()
    roots += [cwd, *cwd.parents, Path('/home/phuc/Documents/HealthSense_rungtamnhi')]
    for r in roots:
        if (r/'experiments/08_v6_locked_protocol/artifacts/V6_RESEARCH_FREEZE_STATUS.json').is_file() and (r/'src/healthsense_ml').is_dir():
            return r.resolve()
    raise FileNotFoundError('Không tìm thấy repo đầy đủ. Clone repo và đặt HEALTHSENSE_REPO=/duong/dan/HealthSense-MachineLearning-Lab hoặc copy notebook vào repo/notebooks/. Không tạo dữ liệu demo thay thế.')

ROOT = find_repo()
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
OUT = Path(os.getenv('HEALTHSENSE_EVIDENCE_OUT', ROOT/'outputs'/'healthsense_defense_evidence')).expanduser()/RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
READ = {}
PLOTS = []

def sha256(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def source(rel, required=True, hash_limit_mb=64):
    p=ROOT/rel
    if not p.is_file():
        if required: raise FileNotFoundError(f'Thiếu artifact bắt buộc: {p}. Không thay bằng số liệu viết tay.')
        READ[rel] = {'status':'MISSING_OPTIONAL'}
        print('CHƯA CÓ local file:',rel)
        return None
    stat=p.stat()
    READ[rel]={'status':'PRESENT','bytes':stat.st_size,'sha256':sha256(p) if stat.st_size<=hash_limit_mb*1024*1024 else 'SKIPPED_LARGE_FILE'}
    return p

def csv(rel, required=True):
    p=source(rel,required)
    return pd.read_csv(p,low_memory=False) if p else None

def js(rel, required=True):
    p=source(rel,required)
    return json.loads(p.read_text(encoding='utf-8')) if p else None

def figure(fig, slug):
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(OUT/f'{slug}.{suffix}',dpi=190,bbox_inches='tight')
    PLOTS.append(slug)
    display(fig)
    plt.close(fig)
    return OUT/f'{slug}.png'

def table(df, slug):
    p=OUT/f'{slug}.csv'
    df.to_csv(p,index=False)
    display(df)
    return p

def git_info():
    def run(*cmd):
        x=subprocess.run(['git','-C',str(ROOT),*cmd],capture_output=True,text=True)
        return x.stdout.strip() if x.returncode==0 else f'UNAVAILABLE: {x.stderr.strip()}'
    return {'head':run('rev-parse','HEAD'),'status_porcelain':run('status','--porcelain')}

def manifest(notebook, notes=None):
    data={'notebook':notebook,'generated_utc':datetime.now(timezone.utc).isoformat(), 'repo':str(ROOT),'git':git_info(),'inputs':READ,'plots':PLOTS,'method':'revisualization of existing artifacts, not model retraining','notes':notes or []}
    p=OUT/f'{notebook}_input_manifest.json'
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Nguồn và checksum:',p)
    print('Biểu đồ:',len(PLOTS),'| thư mục:',OUT)
    return p

def assert_close(a,b,tol=1e-9,msg='metric mismatch'):
    if pd.isna(a) or pd.isna(b) or abs(float(a)-float(b))>tol:
        raise AssertionError(f'{msg}: {a} != {b}')

def confusion_metrics(tp,tn,fp,fn):
    tp,tn,fp,fn=map(int,(tp,tn,fp,fn))
    div=lambda a,b:a/b if b else float('nan')
    sens=div(tp,tp+fn);spec=div(tn,tn+fp)
    prec=div(tp,tp+fp);f1=div(2*tp,2*tp+fp+fn)
    return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':tp+tn+fp+fn,'sensitivity':sens,'specificity':spec,'accuracy':div(tp+tn,tp+tn+fp+fn),'precision':prec,'F1':f1,'balanced_accuracy':(sens+spec)/2}

def metric_from_predictions(df, threshold=.5):
    # Requires the exact frozen prediction schema, no ambiguous guesses.
    required={'label','meta_probability'}
    if not required.issubset(df.columns): raise ValueError(f'Expected {required}; received {df.columns.tolist()}')
    y=pd.to_numeric(df.label,errors='raise').astype(int)
    p=pd.to_numeric(df.meta_probability,errors='raise').astype(float)
    if not y.isin([0,1]).all() or not p.between(0,1).all(): raise ValueError('Expected binary labels and probability in [0,1]')
    pred=p.ge(threshold)
    tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
    return confusion_metrics(tp,tn,fp,fn)

# Matplotlib aesthetics, same blue/green family as project slides.
BLUE='#2464a9';GREEN='#13886a';RED='#bb4b50';ORANGE='#d78a2c';PURPLE='#7654aa'
plt.rcParams.update({'figure.figsize':(9,4.6),'figure.dpi':115,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
