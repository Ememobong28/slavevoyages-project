"""
evaluate.py -- splits and tasks for the imputation experiments.

How a test works (same idea as before, now with a configurable training share):
  * Only rows where the target is KNOWN (in the chosen mode) can be scored.
  * train_frac is the share of those KNOWN values used for training. 0.5 means "half of the
    values we actually have". Because we sample only from known values, training can never end
    up all-missing; if there are not enough known values we stop with a clear message instead.
  * The rest is split into validation (to tune settings) and test (scored once).
  * Rows where the target is truly missing are kept apart as `X_missing`: these are the rows a
    finished method would fill in. They are never scored.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from eval_pipeline import score, baseline_features   # same metrics + feature list as before
import voyage_db

SEED = 42
MIN_TRAIN = 30   # fewer known training values than this -> refuse, results would be noise

@dataclass
class TargetSpec:
    name: str
    kind: str                                   # 'classification' | 'regression'
    drop: list = field(default_factory=list)    # leakage columns removed from the predictors
    exclude_labels: list = field(default_factory=list)
    log_target: bool = False

SPECS = {
 'NATIONAL': TargetSpec('NATIONAL', 'classification',
     drop=['NATIONAL', 'NATIONAL_GROUP', 'TONTYPE', 'NATINIMP', 'XMIMPFLAG', 'TONMOD', 'SLAXIMP', 'SLAMIMP', 'PLACREG', 'REGISREG'],
     exclude_labels=[3, 6]),
 'NATIONAL_GROUP': TargetSpec('NATIONAL_GROUP', 'classification',
     drop=['NATIONAL', 'NATIONAL_GROUP', 'TONTYPE', 'NATINIMP', 'XMIMPFLAG', 'TONMOD', 'SLAXIMP', 'SLAMIMP', 'PLACREG', 'REGISREG']),
 'TONNAGE': TargetSpec('TONNAGE', 'regression',
     drop=['TONNAGE', 'TONMOD', 'SLAXIMP', 'SLAMIMP', 'TSLMTIMP', 'VYMRTIMP', 'VYMRTRAT', 'XMIMPFLAG'], log_target=True),
}

@dataclass
class Task:
    spec: TargetSpec; mode: str; scheme: str; train_frac: float; seed: int
    X_train: pd.DataFrame; y_train: pd.Series
    X_val: pd.DataFrame; y_val: pd.Series
    X_test: pd.DataFrame; y_test: pd.Series
    X_missing: pd.DataFrame          # rows whose target is truly unknown (never scored)
    meta: pd.DataFrame               # VOYAGEID, split, YEARDEP
    notes: list = field(default_factory=list)

def _features(view, spec):
    feats = [c for c in baseline_features() if c not in spec.drop and c in view.columns]
    return view[feats].astype(float)

def make_split(y, spec, order_year, scheme='random', train_frac=0.70, val_share=0.5, seed=SEED, min_train=MIN_TRAIN):
    """Return a Series of 'train'/'val'/'test'/'unknown'/'excluded_code' per row."""
    if not 0 < train_frac < 1:
        raise ValueError(f'train_frac must be strictly between 0 and 1, got {train_frac}')
    if not 0 < val_share < 1:
        raise ValueError('val_share must be strictly between 0 and 1')
    excluded = y.notna() & y.isin(spec.exclude_labels)
    known = y.notna() & ~excluded
    idx = y.index[known]
    n_train = int(round(train_frac * len(idx)))
    if len(idx) == 0:
        raise ValueError(f'{spec.name}: no known values in this mode, nothing to learn from or score')
    if n_train < min_train:
        raise ValueError(f'{spec.name}: train_frac={train_frac} on {len(idx)} known values gives only {n_train} '
                         f'training values (minimum {min_train}). Raise train_frac or pick a target with more known values.')
    if len(idx) - n_train < 2:
        raise ValueError(f'{spec.name}: train_frac={train_frac} leaves fewer than 2 values for validation and test')
    if scheme == 'random':
        strat = None
        if spec.kind == 'classification':
            vc = y[idx].value_counts(); strat = y[idx].where(y[idx].map(vc) >= 10, -1)
            if strat.nunique() > n_train or strat.nunique() > len(idx) - n_train:   # too many classes to stratify
                strat = None
        tr, rest = train_test_split(idx, train_size=n_train, random_state=seed, stratify=strat)
        n_val = max(1, int(round(val_share * len(rest)))); n_val = min(n_val, len(rest) - 1)
        s2 = None
        if strat is not None:
            s2 = strat.loc[rest]; s2 = s2.where(s2.map(s2.value_counts()) >= 2, -1)
            if s2.nunique() > min(n_val, len(rest) - n_val): s2 = None
        va, te = train_test_split(rest, train_size=n_val, random_state=seed, stratify=s2)
    elif scheme == 'temporal':
        order = order_year.loc[idx].sort_values(kind='stable').index
        tr = order[:n_train]; rest = order[n_train:]
        n_val = max(1, int(round(val_share * len(rest)))); n_val = min(n_val, len(rest) - 1)
        va, te = rest[:n_val], rest[n_val:]
    else:
        raise ValueError(f"scheme must be 'random' or 'temporal', got {scheme!r}")
    s = pd.Series('unknown', index=y.index)
    s[tr], s[va], s[te] = 'train', 'val', 'test'
    s[excluded] = 'excluded_code'
    return s

def get_task(base_df, target, mode='real', scheme='random', train_frac=0.70, seed=SEED, val_share=0.5):
    spec = SPECS[target]
    view = voyage_db.build_view(base_df, mode)
    if target not in view or view[target].notna().sum() == 0:
        raise ValueError(f'{target} has no known values in mode={mode!r}')
    y = view[target].astype(float)
    split = make_split(y, spec, base_df['YEARDEP'], scheme, train_frac, val_share, seed)
    X = _features(view, spec)
    if spec.log_target: y = np.log1p(y)
    m = lambda k: split == k
    assert y[m('train')].notna().all() and m('train').sum() > 0, 'training targets must all be known'
    notes = []
    if spec.kind == 'classification':
        unseen = set(y[m('test')].dropna().astype(int)) - set(y[m('train')].astype(int))
        if unseen:
            notes.append(f'{len(unseen)} class(es) in test never appear in training: {sorted(unseen)}')
    meta = base_df[['VOYAGEID', 'YEARDEP']].assign(split=split)
    return Task(spec, mode, scheme, train_frac, seed, X[m('train')], y[m('train')], X[m('val')], y[m('val')],
                X[m('test')], y[m('test')], X[m('unknown')], meta, notes)

def check_integrity(t):
    ids = lambda X: set(t.meta.loc[X.index, 'VOYAGEID'])
    a, b, c = ids(t.X_train), ids(t.X_val), ids(t.X_test)
    assert not (a & b or a & c or b & c), 'a voyage appears in two sets'
    bad = [c for c in t.spec.drop if c in t.X_train.columns]
    assert not bad, f'leakage columns present: {bad}'
    assert t.y_train.notna().all() and t.y_test.notna().all()
    return True
