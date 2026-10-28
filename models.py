"""
models.py -- the model registry.  To add a model, write ONE function and decorate it:

    @register('my_model', kinds=('classification',))
    def my_model(task):
        # fit on task.X_train / task.y_train (tune on task.X_val / task.y_val if you like)
        # return predictions for task.X_test, as an array in the same order
        return predictions            # or: return predictions, 'settings used as text'

Nothing else has to change: run_experiment.py picks it up by name.
Targets are log1p(tonnage) for TONNAGE, so regression predictions are on that scale too.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from evaluate import score

REGISTRY = {}

def register(name, kinds=('classification', 'regression')):
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, kinds=tuple(kinds)); return fn
    return deco

def _tune(task, make, grid, X=lambda d: d):
    """Pick settings on the validation set only (macro-F1 or log-MAE), refit on train."""
    clf = task.spec.kind == 'classification'; key = 'macro_f1' if clf else 'MAE_log'; best = None
    ytr = task.y_train.astype(int) if clf else task.y_train
    for p in grid:
        m = make(**p).fit(X(task.X_train), ytr)
        s = score(task, task.y_val, m.predict(X(task.X_val)))[key]
        if best is None or (s > best[0] if clf else s < best[0]): best = (s, p)
    m = make(**best[1]).fit(X(task.X_train), ytr)
    return m.predict(X(task.X_test)), str(best[1])

@register('majority_or_median')
def reference(task):
    if task.spec.kind == 'classification':
        return np.repeat(task.y_train.value_counts().index[0], len(task.y_test)), 'most common class'
    return np.repeat(task.y_train.median(), len(task.y_test)), 'median'

@register('tree')
def tree(task):
    clf = task.spec.kind == 'classification'
    if clf:
        grid = [dict(max_depth=d, min_samples_leaf=l, class_weight='balanced', random_state=0) for d in (4, 8, 12, None) for l in (1, 10, 30)]
        return _tune(task, DecisionTreeClassifier, grid, X=lambda d: d.fillna(-1))
    grid = [dict(max_depth=d, min_samples_leaf=l, random_state=0) for d in (4, 8, 12, None) for l in (1, 10, 30)]
    return _tune(task, DecisionTreeRegressor, grid, X=lambda d: d.fillna(-1))

@register('gradient_boosting')
def gradient_boosting(task):
    clf = task.spec.kind == 'classification'
    base = dict(max_iter=200, early_stopping=False, random_state=0)
    grid = [dict(max_depth=d, learning_rate=lr, **base) for d in (3, 6) for lr in (0.05, 0.1)]
    if clf:
        return _tune(task, lambda **kw: HistGradientBoostingClassifier(class_weight='balanced', **kw), grid)
    return _tune(task, HistGradientBoostingRegressor, grid)

@register('random_forest')   # example of an added model: it only needed these few lines
def random_forest(task):
    clf = task.spec.kind == 'classification'
    X = lambda d: d.fillna(-1)
    if clf:
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=0).fit(X(task.X_train), task.y_train.astype(int))
    else:
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X(task.X_train), task.y_train)
    return m.predict(X(task.X_test)), 'n_estimators=200, min_samples_leaf=2 (not tuned)'

try:   # TabPFN needs a license key and (ideally) a GPU: it is registered only when installed
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    @register('tabpfn')
    def tabpfn(task):
        if task.spec.kind == 'classification':
            ytr = task.y_train.astype(int)
            try: m = TabPFNClassifier(random_state=0).fit(task.X_train, ytr); note = 'default settings'
            except Exception as e:
                keep = ytr.value_counts().index[:9]; m = TabPFNClassifier(random_state=0).fit(task.X_train, ytr.where(ytr.isin(keep), -1)); note = f'rare classes pooled ({e})'
            return m.predict(task.X_test), note
        return TabPFNRegressor(random_state=0).fit(task.X_train, task.y_train).predict(task.X_test), 'default settings'
except Exception:
    pass
