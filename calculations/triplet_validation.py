"""Stage 2: training diagnostics and evaluation of every frozen candidate."""
from triplet_screener_v2 import (ROOT,SPLIT,HORIZONS,KEYS,aligned,fit,r2,parse_args,digest,load_frame,verify_source)
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
def diagnostics(x, y, model, k, permutations, rng):
    """Training-only diagnostics, not an independent validation or causal test."""
    passed = 0
    windows = max(0, len(y)-252+1)
    for start in range(windows):
        m = fit(x[start:start+252], y[start:start+252])
        passed += int(np.all(m.pvalues[1:] < .05))
    # Circularly shift the whole input pair together to retain A/B dependence.
    # This is a sensitivity diagnostic: nonstationarity can invalidate its null.
    shifts = np.arange(20, len(y)-19)
    actual = model.rsquared
    count = 0
    for shift in rng.choice(shifts, size=permutations, replace=True):
        shuffled = np.roll(x, int(shift), axis=0)
        m = fit(shuffled, y)
        count += int(m.rsquared >= actual)
    robust = model.get_robustcov_results(cov_type='HAC', maxlags=max(5,k))
    return dict(Train_Rolling_Windows=windows,
                Train_Rolling_Sig_Fraction=passed/windows if windows else np.nan,
                Train_Shift_P=(count+1)/(permutations+1),
                Train_HAC_p_A=float(robust.pvalues[1]),
                Train_HAC_p_B=float(robust.pvalues[2]))


def main():
    args=parse_args(__doc__,shuffles=True)
    out=args.output.resolve()
    meta=json.loads((out/'screen_manifest.json').read_text())
    source,frame=load_frame();verify_source(meta,source)
    if digest(Path(__file__).with_name('triplet_screener_v2.py'))!=meta['screening_script_sha256']:
        raise ValueError('Screening code changed; rerun screening.')
    frozen_hash=meta['candidate_sha256']
    if digest(out/'frozen_training_candidates.csv')!=frozen_hash: raise ValueError('Frozen candidate list changed')
    if (out/'validation_manifest.json').exists(): raise SystemExit('Validation already completed in this folder. Use a new screening output directory for a new run.')
    candidates=pd.read_csv(out/'frozen_training_candidates.csv')
    records=[]; predictions=[]
    rng=np.random.default_rng(42)
    for number,(_,row) in enumerate(candidates.iterrows(),1):
        a,b,c,k=row.Stock_A,row.Stock_B,row.Stock_C,int(row.Horizon_k)
        x,y,fd,td,tr,te=aligned(frame,a,b,c,k)
        xt,yt,xo,yo=x[tr],y[tr],x[te],y[te]
        m=fit(xt,yt)
        pred=m.predict(sm.add_constant(xo,has_constant='add'))
        pred_a=fit(xt[:,[0]],yt).predict(sm.add_constant(xo[:,[0]],has_constant='add'))
        pred_b=fit(xt[:,[1]],yt).predict(sm.add_constant(xo[:,[1]],has_constant='add'))
        base=float(yt.mean())
        d=row.to_dict();d.update(diagnostics(xt,yt,m,k,args.shuffles,rng))
        d.update(N_Test=int(te.sum()),Test_First_Feature=str(fd[te][0].date()),
            Test_First_Target=str(td[te][0].date()),Train_Target_Mean=base,
            Test_R2_TrainMean=r2(yo,pred,base),
            Test_R2_TestMean_Descriptive=r2(yo,pred,yo.mean()),
            Test_R2_A_Only=r2(yo,pred_a,base),Test_R2_B_Only=r2(yo,pred_b,base),
            Test_MSE_AB=float(np.mean((yo-pred)**2)),
            Test_MSE_A=float(np.mean((yo-pred_a)**2)),Test_MSE_B=float(np.mean((yo-pred_b)**2)))
        records.append(d)
        predictions.append(pd.DataFrame(dict(Stock_A=a,Stock_B=b,Stock_C=c,Horizon_k=k,
            Feature_Date=fd[te],Target_Date=td[te],Actual=yo,Prediction_AB=pred,
            Prediction_A=pred_a,Prediction_B=pred_b,Training_Mean=base)))
        if number%10==0 or number==len(candidates):
            print(f'Diagnostics and evaluation {number}/{len(candidates)}',flush=True)
    extra=['Train_Rolling_Windows','Train_Rolling_Sig_Fraction','Train_Shift_P','Train_HAC_p_A','Train_HAC_p_B','N_Test','Test_First_Feature','Test_First_Target','Train_Target_Mean','Test_R2_TrainMean','Test_R2_TestMean_Descriptive','Test_R2_A_Only','Test_R2_B_Only','Test_MSE_AB','Test_MSE_A','Test_MSE_B']
    result=pd.DataFrame(records,columns=list(candidates.columns)+extra)
    result.to_csv(out/'candidate_evaluation.csv',index=False)
    pred_columns=KEYS+['Feature_Date','Target_Date','Actual','Prediction_AB','Prediction_A','Prediction_B','Training_Mean']
    (pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=pred_columns)).to_csv(out/'test_predictions.csv',index=False)
    assert hashlib.sha256((out/'frozen_training_candidates.csv').read_bytes()).hexdigest()==frozen_hash

    manifest=dict(source_sha256=digest(source),candidate_sha256=frozen_hash,permutations=args.shuffles,
        validation_script_sha256=digest(__file__),evaluation_sha256=digest(out/'candidate_evaluation.csv'),
        predictions_sha256=digest(out/'test_predictions.csv'))
    (out/'validation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Evaluation complete. Next run compile_results.py.',flush=True)

if __name__=='__main__': main()
