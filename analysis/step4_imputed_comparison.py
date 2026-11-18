"""
Step 4: is the dataset's imputed nationality (NATINIMP) a copy of the recorded one, and does our model agree with it?
 A. copy check: where both NATIONAL and NATINIMP exist, are they the same after the 20 -> 8 group crosswalk?
 B. train on recorded nationality (as 8 groups), predict the 7,377 voyages that have only the dataset's imputed value, compare.
 Run: python analysis/step4_imputed_comparison.py --csv voyages.sqlite --out results/step4
"""
import argparse, os, sys, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import voyage_db, models
from evaluate import get_task

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default='voyages.sqlite'); ap.add_argument('--out', default='results/step4'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); base = voyage_db.read_base(a.csv)
    lab = dict(sqlite3.connect(a.csv).execute("SELECT code, label FROM category_labels WHERE column_name='NATINIMP'").fetchall()) if a.csv.endswith('.sqlite') else {}
    # ---- A. copy check
    grp = base['NATIONAL'].map(voyage_db.NATIONAL_TO_GROUP); imp = base['NATINIMP']; both = grp.notna() & imp.notna()
    same = (grp[both] == imp[both]); diff = base.loc[both][~same][['VOYAGEID', 'YEARDEP', 'NATIONAL', 'NATINIMP']].assign(NATIONAL_as_group=grp[both][~same])
    diff.to_csv(f'{a.out}/copy_check_disagreements.csv', index=False)
    A = dict(both_present=int(both.sum()), same_after_crosswalk=int(same.sum()), different=int((~same).sum()),
             recorded_only=int((base.NATIONAL.notna() & imp.isna()).sum()), imputed_only=int((base.NATIONAL.isna() & imp.notna()).sum()), neither=int((base.NATIONAL.isna() & imp.isna()).sum()))
    print('A. copy check', A); print(diff.to_string())
    # ---- B. model vs imputed
    task = get_task(base, 'NATIONAL_GROUP', 'real', 'random', 0.85, 42); models.REGISTRY['gradient_boosting']['fn'](task); m = task.fitted
    held = accuracy_score(task.y_test.astype(int), m.predict(task.X_test))
    Xm = task.X_missing; idx = Xm.index; has_imp = imp.loc[idx].notna(); Xi = Xm[has_imp.values]
    P = m.predict_proba(Xi); pred = m.classes_[P.argmax(1)]; conf = P.max(1); truth = imp.loc[Xi.index].astype(int).values
    agree = pred == truth; maj = pd.Series(task.y_train.astype(int)).mode()[0]
    out = pd.DataFrame({'VOYAGEID': base.loc[Xi.index, 'VOYAGEID'].values, 'YEARDEP': base.loc[Xi.index, 'YEARDEP'].values, 'imputed_NATINIMP': truth, 'model': pred, 'model_confidence': conf.round(3), 'agree': agree})
    out.to_csv(f'{a.out}/model_vs_imputed_7377.csv', index=False)
    B = dict(n=len(out), agreement=100 * agree.mean(), held_out_accuracy_on_recorded=100 * held, always_majority_agreement=100 * (truth == maj).mean(), majority_group=lab.get(maj, maj))
    print('\nB.', {k: (round(v, 1) if isinstance(v, float) else v) for k, v in B.items()})
    out['bucket'] = pd.cut(out.model_confidence, [0, .6, .8, .95, 1.01], labels=['<60%', '60-80%', '80-95%', '95%+'])
    bc = out.groupby('bucket', observed=True).agg(n=('agree', 'size'), agreement_pct=('agree', lambda s: 100 * s.mean())).round(1); bc.to_csv(f'{a.out}/agreement_by_confidence.csv'); print(bc)
    out['group'] = out.imputed_NATINIMP.map(lambda c: lab.get(c, str(c)))
    bg = out.groupby('group').agg(n=('agree', 'size'), agreement_pct=('agree', lambda s: 100 * s.mean())).round(1).sort_values('n', ascending=False); bg.to_csv(f'{a.out}/agreement_by_group.csv'); print(bg)
    out['period'] = pd.cut(out.YEARDEP, [1500, 1750, 1800, 1850, 1900], labels=['<1750', '1750-99', '1800-49', '1850+'], right=False)
    bp = out.groupby('period', observed=True).agg(n=('agree', 'size'), agreement_pct=('agree', lambda s: 100 * s.mean())).round(1); bp.to_csv(f'{a.out}/agreement_by_period.csv'); print(bp)
    # where they disagree, what is the pattern
    dis = out[~out.agree].groupby(['group']).size().sort_values(ascending=False); 
    pair = out[~out.agree].assign(model_label=lambda d: d.model.map(lambda c: lab.get(c, str(c)))).groupby(['group', 'model_label']).size().sort_values(ascending=False).head(10)
    pair.to_csv(f'{a.out}/top_disagreement_pairs.csv', header=['voyages']); print(pair)
    pd.DataFrame([{**A, **B}]).to_csv(f'{a.out}/summary.csv', index=False)
    order = list(pd.Series(truth).value_counts().index); cm = confusion_matrix(truth, pred, labels=order); cmn = 100 * cm / cm.sum(1, keepdims=True); nm = [lab.get(c, str(c)) for c in order]
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=100); ax.set_xticks(range(len(order)), nm, rotation=40, ha='right'); ax.set_yticks(range(len(order)), nm)
    for i in range(len(order)):
        for j in range(len(order)): ax.text(j, i, f'{cmn[i, j]:.0f}', ha='center', va='center', color='white' if cmn[i, j] > 55 else 'black', fontsize=8)
    ax.set_xlabel('Our model'); ax.set_ylabel("Dataset's imputed value"); ax.set_title(f'7,377 voyages with only an imputed nationality\n(% of each imputed group; agreement {100*agree.mean():.1f}%)'); fig.colorbar(im); fig.tight_layout(); fig.savefig(f'{a.out}/model_vs_imputed.png', dpi=160)
if __name__ == '__main__': main()
