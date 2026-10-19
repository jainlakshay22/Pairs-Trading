"""Stage 1: training-only screen and frozen candidate list."""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, zipfile
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
SPLIT = pd.Timestamp('2023-01-01')
HORIZONS = (1, 2, 3, 5)
KEYS = ['Stock_A', 'Stock_B', 'Stock_C', 'Horizon_k']

def aligned(frame, a, b, c, k):
    """Split on both feature and target dates; never let training labels cross split."""
    x = frame[[a, b]].iloc[:-k].to_numpy()
    y = frame[c].iloc[k:].to_numpy()
    features, targets = frame.index[:-k], frame.index[k:]
    train = (features < SPLIT) & (targets < SPLIT)
    test = (features >= SPLIT) & (targets >= SPLIT)
    return x, y, features, targets, np.asarray(train), np.asarray(test)


def fit(x, y):
    return sm.OLS(y, sm.add_constant(x, has_constant='add')).fit()


def r2(y, pred, baseline):
    den = float(np.sum((y-baseline)**2))
    return float(1-np.sum((y-pred)**2)/den) if den > 0 else np.nan


def parse_args(description, shuffles=False):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--output',type=Path,default=ROOT/'results/training_only')
    if shuffles: parser.add_argument('--shuffles',type=int,default=1000)
    args=parser.parse_args()
    if shuffles and args.shuffles<1: parser.error('--shuffles must be positive')
    return args

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_frame():
    source = ROOT/'data/nse_log_returns.csv'
    frame = pd.read_csv(source,index_col=0,parse_dates=True)
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError('Dates must be sorted and unique')
    if frame.shape[1] != 25 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Expected 25 stocks with finite returns')
    if (frame.index < SPLIT).sum() < 300 or (frame.index >= SPLIT).sum() < 100:
        raise ValueError('Insufficient data around the fixed split')
    return source,frame

def verify_source(meta, source):
    if digest(source)!=meta['source_sha256'] or meta['split']!=str(SPLIT.date()) or meta['horizons']!=list(HORIZONS):
        raise ValueError('Data or split/horizons changed. Rerun screening in a new output directory.')

def main():
    args=parse_args(__doc__)
    out=args.output.resolve()
    if out.exists(): raise SystemExit(f'Output exists: {out}. Use --output with a new directory to preserve previous results.')
    source,frame=load_frame()
    out.mkdir(parents=True)
    rows=[]
    pairs=list(itertools.combinations(frame.columns,2))
    for number,(a,b) in enumerate(pairs,1):
        for c in frame.columns:
            if c in (a,b): continue
            for k in HORIZONS:
                x,y,fd,td,tr,te=aligned(frame,a,b,c,k)
                m=fit(x[tr],y[tr])
                rows.append(dict(Stock_A=a,Stock_B=b,Stock_C=c,Horizon_k=k,
                    N_Train=int(tr.sum()),Alpha=m.params[0],Beta1_A=m.params[1],
                    Beta2_B=m.params[2],Train_p_A=m.pvalues[1],Train_p_B=m.pvalues[2],
                    Train_R2=m.rsquared,Train_Last_Feature=str(fd[tr][-1].date()),
                    Train_Last_Target=str(td[tr][-1].date())))
        if number%25==0: print(f'Screening {number}/{len(pairs)} pairs',flush=True)
    screen=pd.DataFrame(rows)
    # max(p_A,p_B) tests the conjunction that both slopes differ from zero.
    # BY is conservative under arbitrary dependence; p-values retain OLS assumptions.
    screen['Train_Both_P']=screen[['Train_p_A','Train_p_B']].max(axis=1)
    screen['Train_BY_Q']=multipletests(screen.Train_Both_P,method='fdr_by')[1]
    screen['Selected_Train_Only']=(screen.Train_p_A<.05)&(screen.Train_p_B<.05)&(screen.Train_R2>.01)
    screen.to_csv(out/'training_screen_all.csv',index=False)
    candidates=screen[screen.Selected_Train_Only].copy()
    # Freeze selection before any evaluation on later-period returns.
    candidates.to_csv(out/'frozen_training_candidates.csv',index=False)
    frozen_hash=hashlib.sha256((out/'frozen_training_candidates.csv').read_bytes()).hexdigest()

    meta=dict(source_sha256=digest(source),candidate_sha256=frozen_hash,screen_sha256=digest(out/'training_screen_all.csv'),
        split=str(SPLIT.date()),horizons=list(HORIZONS),screened=len(screen),training_selected=len(candidates),
        screening_script_sha256=digest(__file__))
    (out/'screen_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Screened {len(screen)} models; froze {len(candidates)} training-only candidates. Next run triplet_validation.py.',flush=True)

if __name__=='__main__': main()
