"""Run:  python tests/test_framework.py   (no pytest needed; needs tastdb-exp-2019.csv in the repo folder). Each test states the rule it protects."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import voyage_db, evaluate, models
from evaluate import get_task, make_split, check_integrity, SPECS
from run_experiment import run

BASE = voyage_db.read_clean('tastdb-exp-2019.csv')
ok = lambda s: print('PASS', s)

def test_unknown_marker():
    f = voyage_db.load_database(df=BASE, mode='real', marker='UNKNOWN')
    assert not f.isna().any().any(), 'no NaN or empty cell may remain'
    assert not (f.astype(str).apply(lambda c: c.str.strip().isin(['', ',,']))).any().any(), 'no blank-text cells'
    ok('every missing cell is the single symbol UNKNOWN')

def test_modes():
    r = voyage_db.build_view(BASE, 'real'); i = voyage_db.build_view(BASE, 'imputed')
    for c in ['PORTDEP', 'MAJBUYPT', 'MAJSELPT', 'NATIONAL_GROUP', 'YEARDEP']:
        assert i[c].notna().sum() >= r[c].notna().sum(), c
    assert r['NATINIMP'].isna().all() and r['PTDEPIMP'].isna().all(), 'real mode must contain no imputed columns'
    both = r['PORTDEP'].notna()
    assert (r.loc[both, 'PORTDEP'] == i.loc[both, 'PORTDEP']).all(), 'imputed mode must never overwrite a recorded value'
    assert r['NATIONAL'].equals(BASE['NATIONAL']) and i['NATIONAL'].equals(BASE['NATIONAL'])
    ok('real mode has no imputed values; imputed mode only fills gaps and never overwrites')

def test_train_fraction():
    for fr in (0.2, 0.5, 0.8):
        t = get_task(BASE, 'TONNAGE', 'real', 'random', fr); check_integrity(t)
        known = t.y_train.size + t.y_val.size + t.y_test.size
        assert abs(t.y_train.size - round(fr * known)) <= 1, (fr, t.y_train.size, known)
        assert t.y_train.notna().all()
    ok('train_frac is the share of KNOWN values (0.2 / 0.5 / 0.8 checked)')

def test_all_missing_edge_case():
    df = BASE.copy(); keep = df.index[:400]; df.loc[df.index.difference(keep), 'TONNAGE'] = np.nan   # only 400 known
    df.loc[keep[:380], 'TONNAGE'] = np.nan                                                          # now only 20 known
    try:
        get_task(df, 'TONNAGE', 'real', 'random', 0.5); raise SystemExit('should have refused')
    except ValueError as e:
        assert 'training values' in str(e)
    t = get_task(BASE, 'TONNAGE', 'real', 'random', 0.5)
    assert t.y_train.notna().all() and len(t.y_train) > 0
    ok('50% of 20 known values is refused with a clear message; normal 50% never yields missing training labels')

def test_bad_inputs():
    for bad in (0, 1, -0.1, 1.5):
        try: get_task(BASE, 'TONNAGE', 'real', 'random', bad); raise SystemExit(f'accepted {bad}')
        except ValueError: pass
    ok('train_frac outside (0,1) is refused')

def test_temporal_order():
    t = get_task(BASE, 'NATIONAL_GROUP', 'real', 'temporal', 0.6)
    y = t.meta.set_index(t.meta.index)['YEARDEP']
    assert y[t.X_train.index].max() <= y[t.X_test.index].min()
    ok('temporal split: every training voyage is earlier than every test voyage')

def test_new_model_plugs_in():
    @models.register('always_seven', kinds=('classification',))
    def always_seven(task): return np.full(len(task.y_test), 7)
    res, _ = run('NATIONAL_GROUP', 'real', 'random', 0.3, model_names=['majority_or_median', 'always_seven'], base=BASE)
    assert set(res.model) == {'majority_or_median', 'always_seven'}
    ok('a new model is added with one decorated function and appears in the comparison')


def _norm(s):
    if s.dtype.kind in 'fiub': return pd.to_numeric(s, errors='coerce').astype(float).round(6).astype(object).where(s.notna(), '#').astype(str)
    return s.astype(object).where(s.notna(), '#').astype(str)

def test_sql_views_match_python():
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), 'v.sqlite'); voyage_db.build_sqlite('tastdb-exp-2019.csv', path)
    for mode in voyage_db.MODES:
        a = voyage_db.build_view(BASE, mode); s = voyage_db.read_view(path, mode)
        assert set(a.columns) == set(s.columns)
        for c in a.columns:
            b = s[c]
            if a[c].dtype.kind in 'fi': b = pd.to_numeric(b, errors='coerce')
            assert (_norm(a[c]).values == _norm(b).values).all(), (mode, c)
    base2 = voyage_db.read_base(path)
    res, _ = run('NATIONAL_GROUP', 'real', 'random', 0.5, model_names=['majority_or_median', 'tree'], base=base2)
    assert len(res) == 2
    ok('the SQL views (voyages_real / voyages_imputed) give exactly the same tables as the Python code, and experiments run from the .sqlite file')

if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'): f()
    print('all tests passed')
