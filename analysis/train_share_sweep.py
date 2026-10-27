import pandas as pd, voyage_db
from run_experiment import run
base = voyage_db.read_base('tastdb-exp-2019.csv'); out = []
for target in ['NATIONAL_GROUP', 'TONNAGE']:
    for mode in ['real', 'imputed']:
        if target == 'TONNAGE' and mode == 'imputed': continue   # no imputed tonnage twin (TONMOD is a conversion, not a gap-filler)
        for fr in [0.1, 0.3, 0.5, 0.7]:
            res, t = run(target, mode, 'random', fr, model_names=['majority_or_median', 'tree', 'gradient_boosting'], base=base)
            out.append(res); pd.concat(out).to_csv('sweep_results.csv', index=False); print(target, mode, fr, flush=True)
