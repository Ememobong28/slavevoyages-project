"""
Evaluation pipeline for SlaveVoyages 2019 imputation baselines.

Design:
  * Source CSV is never modified. Hidden blanks (whitespace, ',,') are converted to NaN in memory.
  * One TargetSpec per prediction target: which column, task type, which columns leak it.
  * Splits are made ONLY over rows where the target is observed (labels we can score).
    Rows where the target is truly missing are never scored; they stay available as unlabeled rows.
  * Two split schemes: 'random' (stratified for classification) and 'temporal'
    (train on earlier voyages, test on later) to probe non-random missingness / drift.
  * Masking: val/test targets are hidden (set to NaN) in the table a model sees; labels are
    held separately and used only by the scorer.
  * All models must call get_task(...) so they share identical rows, features, and metrics.
"""
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error

SEED = 42
CSV = 'tastdb-exp-2019.csv'
AUDIT = 'column_audit_276.csv'

def load_clean(csv=CSV):
    df = pd.read_csv(csv, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == 'str':
            s = df[c].astype('string').str.strip()
            df[c] = s.mask(s.isin(['', ',,']))
    return df

# columns whose numeric values are category codes (not magnitudes)
CATEGORICAL = ['NATIONAL','TONTYPE','RIG','FATE2','FATE3','FATE4','PORTDEP','EMBPORT','ARRPORT','MAJBUYPT','MAJSELPT',
               'SLA1PORT','PLAC1TRA','PORTRET','PLACCONS','PLACREG','CONSTREG','REGISREG','EMBREG','REGARR','REGEM1',
               'REGDIS1','RETRNREG']

@dataclass
class TargetSpec:
    name: str
    kind: str                       # 'classification' | 'regression'
    drop: list = field(default_factory=list)   # leakage / circular columns
    exclude_labels: list = field(default_factory=list)  # unresolved codes not scored
    log_target: bool = False

SPECS = {
 'NATIONAL': TargetSpec('NATIONAL','classification',
     drop=['NATIONAL','TONTYPE','NATINIMP','XMIMPFLAG','TONMOD','SLAXIMP','SLAMIMP',
           'PLACREG','REGISREG'],   # registration place/region ~ registration country (same event)
     exclude_labels=[3,6]),   # not valid NATIONAL codes per codebook (NATINIMP codes)
 'TONNAGE': TargetSpec('TONNAGE','regression',
     drop=['TONNAGE','TONMOD','SLAXIMP','SLAMIMP','TSLMTIMP','VYMRTIMP','VYMRTRAT','XMIMPFLAG'], log_target=True),
}

def baseline_features(audit=AUDIT):
    a = pd.read_csv(audit)
    keep = a[a.Decision.str.startswith('KEEP')].Column.tolist()
    return [c for c in keep if c not in ('VOYAGEID','YEAR10','YEAR25','YEAR100')]

def build_features(df, spec, feats=None):
    feats = [c for c in (feats or baseline_features()) if c not in spec.drop]
    X = df[feats].copy()
    # drop region column when its place column is the target (place targets later)
    for c in CATEGORICAL:
        if c in X:   # ordinal-encode codes; keeps NaN; trees treat as ordered ids
            X[c] = X[c].astype('float')
    return X

def make_split(df, spec, scheme='random', seed=SEED, frac=(0.70,0.15,0.15)):
    y = df[spec.name]
    ok = y.notna() & ~y.isin(spec.exclude_labels)
    idx = df.index[ok]
    if scheme == 'random':
        strat = None
        if spec.kind == 'classification':
            vc = y[idx].value_counts()
            strat = y[idx].where(y[idx].map(vc) >= 10, -1)   # rare classes pooled for stratification only
        tr, rest = train_test_split(idx, train_size=frac[0], random_state=seed, stratify=strat)
        s2 = strat.loc[rest] if strat is not None else None
        if s2 is not None:   # re-pool classes too small to stratify the second split
            s2 = s2.where(s2.map(s2.value_counts()) >= 2, -1)
        va, te = train_test_split(rest, train_size=frac[1]/(frac[1]+frac[2]), random_state=seed, stratify=s2)
    elif scheme == 'temporal':
        order = df.loc[idx].sort_values(['YEARDEP','VOYAGEID']).index
        n = len(order); a, b = int(n*frac[0]), int(n*(frac[0]+frac[1]))
        tr, va, te = order[:a], order[a:b], order[b:]
    else:
        raise ValueError(scheme)
    s = pd.Series('unlabeled', index=df.index)
    s[tr], s[va], s[te] = 'train', 'val', 'test'
    s[~ok & y.isna()] = 'unlabeled'
    s[y.notna() & y.isin(spec.exclude_labels)] = 'excluded_code'
    return s

@dataclass
class Task:
    spec: TargetSpec; scheme: str
    X_train: pd.DataFrame; y_train: pd.Series
    X_val: pd.DataFrame; y_val: pd.Series
    X_test: pd.DataFrame; y_test: pd.Series
    meta: pd.DataFrame            # VOYAGEID, split, YEARDEP, NATIONAL for subgroup reports
    masked_table: pd.DataFrame    # features + target with val/test target set to NaN

def get_task(df, target, scheme='random', seed=SEED):
    spec = SPECS[target]; split = make_split(df, spec, scheme, seed)
    X = build_features(df, spec); y = df[target].astype(float)
    if spec.log_target: y = np.log1p(y)
    m = lambda k: split == k
    masked = X.copy(); masked[target + '_MASKED'] = y.where(m('train'))
    meta = df[['VOYAGEID','YEARDEP','NATIONAL']].assign(split=split)
    return Task(spec, scheme, X[m('train')], y[m('train')], X[m('val')], y[m('val')], X[m('test')], y[m('test')], meta, masked)

def score(task, y_true, y_pred, subset=None):
    if task.spec.kind == 'classification':
        return dict(n=len(y_true), accuracy=accuracy_score(y_true,y_pred), macro_f1=f1_score(y_true,y_pred,average='macro',zero_division=0),
                    balanced_acc=balanced_accuracy_score(y_true,y_pred))
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if task.spec.log_target:
        return dict(n=len(yt), MAE_tons=mean_absolute_error(np.expm1(yt),np.expm1(yp)),
                    RMSE_tons=mean_squared_error(np.expm1(yt),np.expm1(yp))**0.5,
                    MAE_log=mean_absolute_error(yt,yp), median_abs_err_tons=float(np.median(np.abs(np.expm1(yt)-np.expm1(yp)))))
    return dict(n=len(yt), MAE=mean_absolute_error(yt,yp), RMSE=mean_squared_error(yt,yp)**0.5)

def subgroup_report(task, y_true, y_pred, by='period'):
    """Score by time period (50-year bins) or nationality so models can be compared on easy vs hard slices."""
    m = task.meta.loc[y_true.index]
    key = (m.YEARDEP//50*50).astype(int).astype(str) if by == 'period' else m.NATIONAL.astype('Int64').astype(str)
    rows = []
    for g, ix in key.groupby(key).groups.items():
        if len(ix) >= 20:
            r = score(task, y_true.loc[ix], pd.Series(y_pred, index=y_true.index).loc[ix]); r['group'] = g; rows.append(r)
    return pd.DataFrame(rows).set_index('group')

def save_splits(df, outdir='splits'):
    for t, spec in SPECS.items():
        for sch in ['random','temporal']:
            s = make_split(df, spec, sch)
            pd.DataFrame({'VOYAGEID': df.VOYAGEID, 'split': s}).to_csv(f'{outdir}/split_{t}_{sch}.csv', index=False)

def check_integrity(task):
    """Assertions that guard fair comparison. Raises if violated."""
    tr, va, te = task.X_train.index, task.X_val.index, task.X_test.index
    assert not (set(tr)&set(va)) and not (set(tr)&set(te)) and not (set(va)&set(te)), 'splits overlap'
    assert task.spec.name not in task.X_train.columns, 'target in features'
    assert not (set(task.spec.drop) & set(task.X_train.columns)), 'leakage column present'
    assert task.y_train.notna().all() and task.y_val.notna().all() and task.y_test.notna().all()
    assert task.masked_table.loc[va.union(te), task.spec.name + '_MASKED'].isna().all(), 'val/test labels not masked'
    if task.scheme == 'temporal':
        yr = task.meta.YEARDEP
        assert yr[tr].max() <= yr[te].min(), 'temporal split leaks future into train'
    return True

# ---- readable labels (reporting only; models should use the integer codes) ----
import json, os
def decode(series, col, maps='codebook_maps.json'):
    """Map integer codes to codebook labels for confusion matrices / error analysis. Unknown codes -> 'code <n>'."""
    M = json.load(open(maps)); m = M['categorical'].get(col) or M['geographic']
    return series.map(lambda v: m.get(str(int(v)), f'code {int(v)}') if pd.notna(v) else v)
