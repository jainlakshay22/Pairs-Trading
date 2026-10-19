"""Stage 3: summarize all candidates and save result files, without test-based selection."""
from triplet_screener_v2 import (SPLIT,HORIZONS,parse_args,digest,load_frame,verify_source)
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import scipy,statsmodels

def main():
    args=parse_args(__doc__);out=args.output.resolve()
    source,frame=load_frame()
    meta=json.loads((out/'screen_manifest.json').read_text());verify_source(meta,source)
    vm=json.loads((out/'validation_manifest.json').read_text())
    frozen_hash=digest(out/'frozen_training_candidates.csv')
    for file,key,manifest in [('training_screen_all.csv','screen_sha256',meta),('frozen_training_candidates.csv','candidate_sha256',meta),('candidate_evaluation.csv','evaluation_sha256',vm),('test_predictions.csv','predictions_sha256',vm)]:
        if digest(out/file)!=manifest[key]: raise ValueError(f'{file} changed since its preceding stage')
    if vm['source_sha256']!=meta['source_sha256'] or vm['candidate_sha256']!=frozen_hash:
        raise ValueError('Screening and validation provenance do not match')
    for name,key,manifest in [('triplet_screener_v2.py','screening_script_sha256',meta),('triplet_validation.py','validation_script_sha256',vm)]:
        if digest(Path(__file__).with_name(name))!=manifest[key]:raise ValueError(f'{name} changed since execution')
    screen=pd.read_csv(out/'training_screen_all.csv');candidates=pd.read_csv(out/'frozen_training_candidates.csv');result=pd.read_csv(out/'candidate_evaluation.csv')
    keys=['Stock_A','Stock_B','Stock_C','Horizon_k']
    if not candidates[keys].equals(result[keys]):raise ValueError('Evaluation must contain every frozen candidate in order')
    summary=dict(split=str(SPLIT.date()),horizons=list(HORIZONS),screened=len(screen),
        training_selected=len(candidates),selection_rule='Both unrounded training OLS p < 0.05 and training R2 > 0.01; no test-based filtering',
        training_BY_q_below_005=int((candidates.Train_BY_Q<.05).sum()),
        training_rolling_pass=int((result.Train_Rolling_Sig_Fraction>=.6).sum()) if len(result) else 0,
        training_shift_pass=int((result.Train_Shift_P<.05).sum()) if len(result) else 0,
        positive_test_R2_train_mean=int((result.Test_R2_TrainMean>0).sum()) if len(result) else 0,
        positive_test_R2_test_mean_descriptive=int((result.Test_R2_TestMean_Descriptive>0).sum()) if len(result) else 0,
        beats_both_single_inputs=int(((result.Test_MSE_AB<result.Test_MSE_A)&(result.Test_MSE_AB<result.Test_MSE_B)).sum()) if len(result) else 0,
        permutations=vm['permutations'],seed=42,data_first=str(frame.index.min().date()),data_last=str(frame.index.max().date()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_sha256=frozen_hash,
        script_sha256={name:digest(Path(__file__).with_name(name)) for name in ['triplet_screener_v2.py','triplet_validation.py','compile_results.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),
        limitations=['Retrospective evaluation: 2023-2025 has already been inspected in earlier work.',
        'Existing input data was filled by the legacy downloader; original missingness and adjustment audit remains outstanding.',
        'Nominal screening is exploratory; BY-adjusted results are reported separately.',
        'Circular-shift diagnostic assumes a suitable time-series null and is not proof of causality or regime stability.',
        'No models are declared validated using test performance. No trading metrics are produced.'])

    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # No selection or reordering by test performance.
    result.to_csv(out/'final_results_table.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Upload these files from {out}:',flush=True)
    for name in ['summary.json','final_results_table.csv','training_screen_all.csv','test_predictions.csv']:
        print(f'  {name}',flush=True)

if __name__=='__main__': main()
