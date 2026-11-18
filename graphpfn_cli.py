"""
Command-line runner for GraphPFN (used by the Colab notebook, which runs it inside a Python 3.11 environment).
  python graphpfn_cli.py --targets TONNAGE --schemes random --seeds 1 --graphs none,ship10,ship10_owner_captain
"""
import argparse, os
import pandas as pd
import voyage_db
from graphpfn_run import run_graphpfn

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='voyages.sqlite'); ap.add_argument('--out', default='results/graphpfn')
ap.add_argument('--targets', default='TONNAGE'); ap.add_argument('--schemes', default='random')
ap.add_argument('--seeds', type=int, default=1, help='random splits: seeds 42, 43, ... (the time-based split always uses 42)')
ap.add_argument('--graphs', default='none,ship10,ship10_owner_captain'); ap.add_argument('--finetune', action='store_true')
ap.add_argument('--train-frac', type=float, default=0.70)
a = ap.parse_args(); base = voyage_db.read_base(a.csv); os.makedirs(a.out, exist_ok=True); rows = []
for target in a.targets.split(','):
    for scheme in a.schemes.split(','):
        for seed in range(42, 42 + (a.seeds if scheme == 'random' else 1)):
            for graph in a.graphs.split(','):
                print(f'--- {target} {scheme} seed {seed} graph {graph}', flush=True)
                rows.append(run_graphpfn(target, scheme, a.train_frac, seed, graph, a.finetune, 'real', a.csv, base, a.out))
                pd.concat(rows, ignore_index=True).to_csv(f'{a.out}/all_runs.csv', index=False)
res = pd.concat(rows, ignore_index=True)
cols = [c for c in ['median_APE_pct', 'within_25_pct', 'MAE_tons', 'accuracy', 'macro_f1'] if c in res]
print(res.groupby(['target', 'split', 'model'])[cols].mean().round(3).to_string())
print(res[['model', 'settings', 'seconds']].to_string(index=False))
