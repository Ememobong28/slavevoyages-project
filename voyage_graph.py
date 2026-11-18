"""
voyage_graph.py -- connect voyages to each other, for graph models such as GraphPFN.

A voyage is a node. Two voyages are linked when a rule says they probably share the thing we are predicting:
the same ship (by name) within some years, the same first owner, the same first captain.
Links are never made through the target column, so the graph does not carry the answer.

  build_edges(base, 'ship10_owner_captain')  ->  (src, dst)   both directions, no self links, duplicates removed
  graph_inputs(base, 'TONNAGE', ...)         ->  everything a graph model needs, with test and unknown targets hidden
"""
import numpy as np, pandas as pd
import voyage_db
from evaluate import get_task, _features

# rule = (column, window in years or None).  Neighbours = the K nearest voyages in time inside the same group.
RULES = {
    'ship10':   [('SHIPNAME', 10)],
    'ship30':   [('SHIPNAME', 30)],
    'owner10':  [('OWNERA', 10)],
    'captain10': [('CAPTAINA', 10)],
    'ship10_owner_captain': [('SHIPNAME', 10), ('OWNERA', 10), ('CAPTAINA', 10)],
    'none':     [],
}
K_NEAREST = 10

def _norm(s): return s.astype('string').str.strip().str.upper()

def rule_edges(base, column, window, k=K_NEAREST):
    key = _norm(base[column]); year = base['YEARDEP'].astype(float)
    df = pd.DataFrame({'key': key, 'year': year, 'pos': np.arange(len(base))}).dropna(subset=['key', 'year'])
    src, dst = [], []
    for _, g in df.groupby('key', sort=False):
        if len(g) < 2: continue
        g = g.sort_values('year', kind='stable'); pos = g.pos.values; yr = g.year.values; m = len(g)
        for off in range(1, min(k, m - 1) + 1):
            ok = (yr[off:] - yr[:-off]) <= (window if window is not None else np.inf)
            src.append(pos[:-off][ok]); dst.append(pos[off:][ok])
    if not src: return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(src), np.concatenate(dst)

def build_edges(base, graph='ship10_owner_captain'):
    """Undirected edges as two directed arrays. Unknown rule names raise."""
    if graph not in RULES: raise ValueError(f'unknown graph {graph!r}; choose from {sorted(RULES)}')
    S, D = [], []
    for column, window in RULES[graph]:
        s, d = rule_edges(base, column, window); S.append(s); D.append(d)
    if not S: return np.zeros(0, int), np.zeros(0, int)
    s, d = np.concatenate(S), np.concatenate(D)
    a, b = np.minimum(s, d), np.maximum(s, d); keep = a != b
    e = np.unique(np.stack([a[keep], b[keep]], 1), axis=0)
    return np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])

def graph_inputs(base, target, mode='real', scheme='random', train_frac=0.70, seed=42, graph='ship10_owner_captain'):
    """Same split as every other model (get_task). Test and unknown voyages stay in the graph as nodes with their features,
    but their target is NaN, so the model cannot see the answer."""
    task = get_task(base, target, mode, scheme, train_frac, seed)
    view = voyage_db.build_view(base, mode); X = _features(view, task.spec)
    n = len(base); pos = pd.Series(np.arange(n), index=base.index)
    y_all = np.full(n, np.nan, dtype=np.float32); classes = None
    if task.spec.kind == 'classification':
        classes = np.sort(task.y_train.astype(int).unique()); code = {c: i for i, c in enumerate(classes)}
        ok = lambda s: s.astype(int).isin(code)
        mp = lambda s: s.astype(int).map(code).astype(np.float32).values
        for part in (task.y_train, task.y_val):
            keep = ok(part); y_all[pos[part.index[keep]].values] = mp(part[keep])
    else:
        for part in (task.y_train, task.y_val): y_all[pos[part.index].values] = part.values.astype(np.float32)
    masks = {k: np.zeros(n, bool) for k in ('train', 'val', 'test')}
    for k, part in (('train', task.y_train), ('val', task.y_val), ('test', task.y_test)): masks[k][pos[part.index].values] = True
    src, dst = build_edges(base, graph)
    return dict(task=task, X=X.loc[base.index], targets=y_all, masks=masks, src=src, dst=dst, classes=classes, n=n, graph=graph)
