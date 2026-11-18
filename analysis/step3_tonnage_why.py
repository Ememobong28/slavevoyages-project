"""
Step 3: why does tonnage predict poorly?
 A. error by true ship size   B. error by period   C. error by how many helper columns a voyage has
 D. how much tonnage itself is missing, by period   E. does tonnage drift over time (why the time-based split fails)
Run: python analysis/step3_tonnage_why.py --csv voyages.sqlite --out results/step3
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import voyage_db, models
from evaluate import get_task

HELPERS = ['RIG', 'GUNS', 'TONTYPE', 'SLINTEND', 'TSLAVESD', 'SLAARRIV', 'YRCONS', 'YRREG']

def ape(task, pred): 
    yt = np.expm1(task.y_test.values); return yt, np.expm1(pred), 100 * np.abs(np.expm1(pred) - yt) / yt

def summarize(df, by):
    g = df.groupby(by, observed=True)
    return pd.DataFrame({'n': g.size(), 'median_pct_error': g.pct.median(), 'within_25': 100 * g.pct.apply(lambda s: (s <= 25).mean()),
                         'avg_miss_tons': g.miss.mean(), 'mean_signed_pct': g.signed.mean()}).round(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default='voyages.sqlite'); ap.add_argument('--out', default='results/step3'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); base = voyage_db.read_base(a.csv); res = {}
    for scheme in ['random', 'temporal']:
        task = get_task(base, 'TONNAGE', 'real', scheme, 0.70, 42); models.REGISTRY['gradient_boosting']['fn'](task)
        pred = task.fitted.predict(task.X_test); yt, yp, pct = ape(task, pred); X = task.X_test
        d = pd.DataFrame({'true': yt, 'pred': yp, 'pct': pct, 'miss': np.abs(yp - yt), 'signed': 100 * (yp - yt) / yt, 'year': X['YEARDEP'].values})
        d['size'] = pd.cut(d.true, [0, 60, 100, 150, 250, 400, 5000], labels=['<60', '60-100', '100-150', '150-250', '250-400', '400+'])
        d['period'] = pd.cut(d.year, [1500, 1700, 1750, 1800, 1825, 1850, 1900], labels=['<1700', '1700-49', '1750-99', '1800-24', '1825-49', '1850+'], right=False)
        d['n_helpers'] = X[[c for c in HELPERS if c in X]].notna().sum(axis=1).values
        d['helpers_group'] = pd.cut(d.n_helpers, [-1, 2, 4, 6, 8], labels=['0-2', '3-4', '5-6', '7-8'])
        for name in ['size', 'period', 'helpers_group']:
            t = summarize(d, name); t.to_csv(f'{a.out}/error_by_{name}_{scheme}.csv'); res[(scheme, name)] = t
            print(f'\n== {scheme}: error by {name}\n', t.to_string())
        d.to_csv(f'{a.out}/test_errors_{scheme}.csv', index=False)
    # D, E: from the whole table
    t = base[['YEARDEP', 'TONNAGE'] + [c for c in HELPERS if c in base]].copy()
    t['period'] = pd.cut(t.YEARDEP, [1500, 1700, 1750, 1800, 1825, 1850, 1900], labels=['<1700', '1700-49', '1750-99', '1800-24', '1825-49', '1850+'], right=False)
    miss = t.groupby('period', observed=True).apply(lambda g: pd.Series({'voyages': len(g), 'tonnage_missing_pct': 100 * g.TONNAGE.isna().mean(),
              **{f'{c}_missing_pct': 100 * g[c].isna().mean() for c in HELPERS if c in g}, 'median_tonnage': g.TONNAGE.median(), 'sd_tonnage': g.TONNAGE.std()})).round(1)
    miss.to_csv(f'{a.out}/missingness_and_drift_by_period.csv'); print('\n== missingness / drift by period\n', miss.to_string())
    # plots
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, lab) in enumerate([('size', 'True ship size (tons)'), ('period', 'Period of departure'), ('helpers_group', 'Helper columns known (of 8)')]):
        for sch, c in [('random', '#2a6f97'), ('temporal', '#c1666b')]:
            t = res[(sch, name)]; ax[i].plot(range(len(t)), t.median_pct_error, 'o-', color=c, label=sch + ' split')
        ax[i].set_xticks(range(len(t)), t.index, rotation=30); ax[i].set_xlabel(lab); ax[i].set_ylabel('Median percent error'); ax[i].grid(alpha=.3)
    ax[0].legend(); fig.suptitle('TONNAGE: where the error is'); fig.tight_layout(); fig.savefig(f'{a.out}/error_breakdown.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.plot(range(len(miss)), miss.median_tonnage, 'o-', label='median tonnage (tons)'); ax.plot(range(len(miss)), miss.tonnage_missing_pct, 's--', label='% of voyages with tonnage missing')
    ax.set_xticks(range(len(miss)), miss.index, rotation=30); ax.legend(); ax.grid(alpha=.3); ax.set_title('Tonnage drifts and goes missing over time'); fig.tight_layout(); fig.savefig(f'{a.out}/drift.png', dpi=160)
if __name__ == '__main__': main()
