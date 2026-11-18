"""
Step 2: which columns drive the predictions, and by how much?   (nationality: predicted well. tonnage: predicted poorly.)

For each target (recorded values only, random split, 70% of known values for training):
  1. Fit gradient boosting (settings tuned on validation, as in run_experiment).
  2. PERMUTATION importance on the test set: shuffle one column, see how much the score gets worse.
       nationality -> drop in accuracy (percentage points)
       tonnage     -> rise in the typical (median) percent error (percentage points)
     Repeated over 3 random splits x 10 shuffles; mean and spread are reported.
  3. The same for GROUPS of columns shuffled together (ship size, ports, time ...), because columns in a group carry overlapping information.
  4. Leave-a-group-out: retrain WITHOUT the group and see what is left (one split).
  5. A decision tree's own importance as a second opinion.
  6. Plots: top columns, nationality confusion matrix, tonnage predicted vs true.
Run:  python analysis/step2_feature_importance.py --csv voyages.sqlite --out results/step2
"""
import argparse, os, sys, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import voyage_db, models
from evaluate import get_task

GROUPS = {
 'Ship size and type (rig, tonnage, guns)': ['RIG', 'TONNAGE', 'TONTYPE', 'GUNS'],
 'Ship build and registration (year, place)': ['YRCONS', 'YRREG', 'CONSTREG', 'PLACCONS', 'PLACREG', 'REGISREG'],
 'Year of the voyage': ['YEARDEP', 'YEAR5'],
 'Departure and purchase ports/regions': ['PORTDEP', 'EMBPORT', 'EMBREG', 'PLAC1TRA', 'REGEM1', 'MAJBUYPT'],
 'Landing and return ports/regions': ['ARRPORT', 'REGARR', 'MAJSELPT', 'SLA1PORT', 'REGDIS1', 'PORTRET', 'RETRNREG'],
 'Captives (numbers)': ['SLINTEND', 'TSLAVESD', 'SLAARRIV'],
 'Voyage outcome (FATE2/3/4)': ['FATE2', 'FATE3', 'FATE4'],
 'Nationality of the ship': ['NATIONAL'],
}

def metric(kind, task, model, X, y):
    p = model.predict(X)
    if kind == 'classification':
        return dict(accuracy=100 * accuracy_score(y, p), macro_f1=100 * f1_score(y, p, average='macro', zero_division=0))
    ape = 100 * np.abs(np.expm1(p) - np.expm1(y)) / np.expm1(y)
    return dict(median_pct_error=float(np.median(ape)), within_25=100 * float((ape <= 25).mean()))

HEADLINE = {'classification': 'accuracy', 'regression': 'median_pct_error'}
def worse(kind, base, new):          # how much WORSE the headline metric got (positive = worse)
    k = HEADLINE[kind]
    return (base[k] - new[k]) if kind == 'classification' else (new[k] - base[k])

def permutation(task, model, kind, cols_sets, repeats=10, seed=0):
    rng = np.random.default_rng(seed); X, y = task.X_test, task.y_test
    y = y.astype(int) if kind == 'classification' else y
    base = metric(kind, task, model, X, y); out = {}
    for name, cols in cols_sets.items():
        d = []
        for _ in range(repeats):
            idx = rng.permutation(len(X)); Xp = X.copy(); Xp[cols] = X[cols].values[idx]
            d.append(worse(kind, base, metric(kind, task, model, Xp, y)))
        out[name] = np.mean(d)
    return out, base

def refit(task, kind, cols_drop):
    keep = [c for c in task.X_train.columns if c not in cols_drop]
    mk = (lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=200, early_stopping=False, class_weight='balanced', random_state=0)) \
        if kind == 'classification' else (lambda: HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=200, early_stopping=False, random_state=0))
    ytr = task.y_train.astype(int) if kind == 'classification' else task.y_train
    m = mk().fit(task.X_train[keep], ytr)
    yte = task.y_test.astype(int) if kind == 'classification' else task.y_test
    return metric(kind, task, m, task.X_test[keep], yte)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default='voyages.sqlite'); ap.add_argument('--out', default='results/step2')
    ap.add_argument('--seeds', type=int, default=3); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    base_df = voyage_db.read_base(a.csv)
    con = sqlite3.connect(a.csv) if a.csv.endswith('.sqlite') else None
    summary = []
    for target, kind in [('NATIONAL', 'classification'), ('TONNAGE', 'regression')]:
        per_col, per_grp, bases = [], [], []
        for s in range(42, 42 + a.seeds):
            task = get_task(base_df, target, 'real', 'random', 0.70, s)
            models.REGISTRY['gradient_boosting']['fn'](task); model = task.fitted
            cols = list(task.X_test.columns)
            imp, b = permutation(task, model, kind, {c: [c] for c in cols}, seed=s)
            grp = {g: [c for c in v if c in cols] for g, v in GROUPS.items()}; grp = {g: v for g, v in grp.items() if v}
            gimp, _ = permutation(task, model, kind, grp, seed=s)
            per_col.append(pd.Series(imp)); per_grp.append(pd.Series(gimp)); bases.append(b)
            if s == 42: first = (task, model, grp, b)
        col = pd.concat(per_col, axis=1); grpd = pd.concat(per_grp, axis=1)
        tbl = pd.DataFrame({'worse_when_shuffled_mean': col.mean(axis=1), 'sd_across_splits': col.std(axis=1)}).sort_values('worse_when_shuffled_mean', ascending=False)
        tbl['missing_in_train_pct'] = [100 * first[0].X_train[c].isna().mean() for c in tbl.index]
        tbl.round(3).to_csv(f'{a.out}/importance_columns_{target}.csv')
        gt = pd.DataFrame({'worse_when_shuffled_mean': grpd.mean(axis=1), 'sd_across_splits': grpd.std(axis=1)}).sort_values('worse_when_shuffled_mean', ascending=False)
        # leave-a-group-out, split 42 only
        task, model, grp, b = first
        left = {'(nothing removed)': b[HEADLINE[kind]]}
        for g, v in grp.items(): left[f'without: {g}'] = refit(task, kind, v)[HEADLINE[kind]]
        gt['score_when_group_removed_retrained'] = [left[f'without: {g}'] for g in gt.index]
        gt.round(3).to_csv(f'{a.out}/importance_groups_{target}.csv')
        tree = models.REGISTRY['tree']['fn'](task); tm = task.fitted
        ti = pd.Series(tm.feature_importances_, index=task.X_train.columns).sort_values(ascending=False)
        ti.round(4).to_csv(f'{a.out}/tree_importance_{target}.csv', header=['tree_impurity_importance'])
        rho = pd.concat([tbl.worse_when_shuffled_mean, ti], axis=1).corr(method='spearman').iloc[0, 1]
        summary.append(dict(target=target, baseline=left['(nothing removed)'], metric=HEADLINE[kind], tree_vs_boosting_rank_agreement=round(float(rho), 2)))
        # ---- plots -------------------------------------------------------------------------------------------
        top = tbl.head(12)[::-1]; fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(top.index, top.worse_when_shuffled_mean, xerr=top.sd_across_splits, color='#2a6f97')
        ax.set_xlabel('How much worse the score gets when the column is shuffled\n' + ('(accuracy points)' if kind == 'classification' else '(points of typical percent error)'))
        ax.set_title(f'{target}: what the model relies on'); fig.tight_layout(); fig.savefig(f'{a.out}/importance_{target}.png', dpi=160); plt.close(fig)
        pred = model.predict(task.X_test)
        if kind == 'classification':
            lab = dict(con.execute("SELECT code, label FROM category_labels WHERE column_name='NATIONAL'").fetchall()) if con else {}
            y = task.y_test.astype(int).values; p = pred.astype(int); keep = pd.Series(y).value_counts().index[:7]
            m = np.isin(y, keep) & np.isin(p, keep); order = list(keep)
            cm = confusion_matrix(y[m], p[m], labels=order); cmn = 100 * cm / cm.sum(axis=1, keepdims=True)
            names = [lab.get(c, str(c)) for c in order]; fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=100); ax.set_xticks(range(len(order)), names, rotation=40, ha='right'); ax.set_yticks(range(len(order)), names)
            for i in range(len(order)):
                for j in range(len(order)): ax.text(j, i, f'{cmn[i, j]:.0f}', ha='center', va='center', color='white' if cmn[i, j] > 55 else 'black', fontsize=8)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('NATIONAL: confusion matrix (% of each true nationality)\n7 most common nationalities, test set')
            fig.colorbar(im, label='%'); fig.tight_layout(); fig.savefig(f'{a.out}/confusion_NATIONAL.png', dpi=160); plt.close(fig)
        else:
            yt, yp = np.expm1(task.y_test.values), np.expm1(pred); fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.scatter(yt, yp, s=5, alpha=0.25, color='#2a6f97'); lim = [10, 2000]
            ax.plot(lim, lim, 'k-', lw=1, label='perfect'); ax.plot(lim, [1.25 * v for v in lim], 'k--', lw=0.8, label='25% too high / too low'); ax.plot(lim, [0.75 * v for v in lim], 'k--', lw=0.8)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('True tonnage (tons)'); ax.set_ylabel('Predicted tonnage (tons)')
            ax.set_title('TONNAGE: predicted vs true (test set)'); ax.legend(); fig.tight_layout(); fig.savefig(f'{a.out}/scatter_TONNAGE.png', dpi=160); plt.close(fig)
        print('\n==', target, '| baseline', {k: round(v, 2) for k, v in b.items()}); print(tbl.head(10).round(2).to_string()); print(gt.round(2).to_string())
    pd.DataFrame(summary).to_csv(f'{a.out}/summary.csv', index=False); print(pd.DataFrame(summary))

if __name__ == '__main__':
    main()
