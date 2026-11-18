"""
graphpfn_run.py -- run GraphPFN on the SlaveVoyages task, scored exactly like the other models.
Needs a CUDA GPU and the packages in requirements-graphpfn.txt (use the Colab notebook notebooks/run_graphpfn_colab.ipynb).

  from graphpfn_run import run_graphpfn
  res = run_graphpfn('TONNAGE', 'random', seed=42, graph='ship10_owner_captain', mode_ft=False, csv='voyages.sqlite', outdir='results/graphpfn')

graph = 'none' runs GraphPFN with no links, a fair "graph off" comparison.
"""
import os, time
import numpy as np, pandas as pd
import voyage_db
from evaluate import score
from voyage_graph import graph_inputs

def _predict(gi, device, finetune, fill):
    import dgl, torch
    from functools import partial
    from graphpfn import GraphDataset, predict_icl
    X = gi['X'].copy()
    if fill: X = X.fillna(-1.0)
    g = dgl.graph((torch.tensor(gi['src']), torch.tensor(gi['dst'])), num_nodes=gi['n'])
    kind = gi['task'].spec.kind
    ds = GraphDataset(name='voyages', graph=g, features={'num': X.values.astype(np.float32)}, targets=gi['targets'],
                      masks=gi['masks'], task_type='multiclass' if kind == 'classification' else 'regression')
    if finetune:
        from graphpfn import tune_and_predict_ft
        from graphpfn.inference.util import compute_score, compute_loss
        preds, best = tune_and_predict_ft(dataset=ds, score_fn=partial(compute_score, task_type=ds.task_type),
                                          loss_fn=partial(compute_loss, task_type=ds.task_type), device=device)
        return preds, f'finetune {best}'
    return predict_icl(dataset=ds, device=device), 'in-context'

def run_graphpfn(target, scheme='random', train_frac=0.70, seed=42, graph='ship10_owner_captain', finetune=False, mode='real',
                 csv='voyages.sqlite', base=None, outdir=None, device='cuda:0'):
    base = base if base is not None else voyage_db.read_base(csv)
    gi = graph_inputs(base, target, mode, scheme, train_frac, seed, graph); task = gi['task']
    t0 = time.time(); how = ''
    try: preds, how = _predict(gi, device, finetune, fill=False)
    except Exception as e:                                         # some builds reject NaN features: retry with -1 for missing
        print('NaN features rejected (', type(e).__name__, ') -> retrying with missing filled as -1'); preds, how = _predict(gi, device, finetune, fill=True); how += ', NaN filled -1'
    preds = np.asarray(preds); pos = pd.Series(np.arange(gi['n']), index=base.index)[task.y_test.index].values
    p = preds[pos]
    if task.spec.kind == 'classification': p = gi['classes'][p.argmax(1)] if p.ndim == 2 else gi['classes'][p.astype(int)]
    r = score(task, task.y_test, p)
    name = f'graphpfn_{graph}' + ('_ft' if finetune else '')
    r.update(target=target, mode=mode, split=scheme, train_frac=train_frac, seed=seed, model=name, n_train=len(task.y_train),
             settings=f'{how}; {len(gi["src"])//2} links', seconds=round(time.time() - t0, 1))
    res = pd.DataFrame([r])
    if outdir:
        os.makedirs(outdir, exist_ok=True); tag = f'{target}_{mode}_{scheme}_{int(train_frac*100)}_seed{seed}_{name}'
        res.to_csv(f'{outdir}/results_{tag}.csv', index=False)
        truth = np.expm1(task.y_test.values) if task.spec.log_target else task.y_test.values
        out = pd.DataFrame({'VOYAGEID': task.meta.loc[task.y_test.index, 'VOYAGEID'].values, 'y_true': truth, name: np.expm1(p) if task.spec.log_target else p})
        out.to_csv(f'{outdir}/predictions_{tag}.csv', index=False)
    return res
