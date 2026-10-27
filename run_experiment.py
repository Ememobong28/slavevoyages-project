"""
run_experiment.py -- run any registered models on a target and compare them.

  python run_experiment.py --target NATIONAL_GROUP --mode real --train-frac 0.5 --models tree,gradient_boosting
  python run_experiment.py --target TONNAGE --mode imputed --scheme temporal

Or from Python:  from run_experiment import run ; run('TONNAGE', mode='real', train_frac=0.5)
"""
import argparse, os, time
import pandas as pd
import voyage_db, models
from evaluate import get_task, check_integrity, SPECS
from models import REGISTRY
from evaluate import score

def run(target, mode='real', scheme='random', train_frac=0.70, seed=42, model_names=None, csv='tastdb-exp-2019.csv', base=None, outdir=None):
    base = base if base is not None else voyage_db.read_base(csv)
    task = get_task(base, target, mode, scheme, train_frac, seed); check_integrity(task)
    names = model_names or [n for n, r in REGISTRY.items() if task.spec.kind in r['kinds']]
    rows = []
    for n in names:
        if n not in REGISTRY: raise KeyError(f'unknown model {n!r}; available: {sorted(REGISTRY)}')
        if task.spec.kind not in REGISTRY[n]['kinds']: continue
        t0 = time.time(); out = REGISTRY[n]['fn'](task)
        pred, params = out if isinstance(out, tuple) else (out, '')
        r = score(task, task.y_test, pred)
        r.update(target=target, mode=mode, split=scheme, train_frac=train_frac, seed=seed, model=n, n_train=len(task.y_train),
                 settings=params, seconds=round(time.time() - t0, 1))
        rows.append(r)
    res = pd.DataFrame(rows)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res.to_csv(f'{outdir}/results_{target}_{mode}_{scheme}_{int(train_frac*100)}.csv', index=False)
    return res, task

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='NATIONAL_GROUP', choices=sorted(SPECS))
    ap.add_argument('--mode', default='real', choices=voyage_db.MODES)
    ap.add_argument('--scheme', default='random', choices=['random', 'temporal'])
    ap.add_argument('--train-frac', type=float, default=0.70)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default=None, help='comma list; default = all registered for this target type')
    ap.add_argument('--csv', default='tastdb-exp-2019.csv', help='the original csv OR the voyages.sqlite built by voyage_db.py'); ap.add_argument('--out', default='results')
    a = ap.parse_args()
    res, task = run(a.target, a.mode, a.scheme, a.train_frac, a.seed, a.models.split(',') if a.models else None, a.csv, outdir=a.out)
    for n in task.notes: print('NOTE:', n)
    print(res.drop(columns=['seed']).round(3).to_string(index=False))
