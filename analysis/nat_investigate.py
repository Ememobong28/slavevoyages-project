import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from eval_pipeline import *
df = load_clean()
nat = df.NATIONAL
print('NATIONAL missing: %d of %d (%.2f%%)' % (nat.isna().sum(), len(df), 100*nat.isna().mean()))
print('NATIONAL missing, among rows where NATINIMP present:', df.loc[nat.isna(),'NATINIMP'].notna().sum())
k = get_task(df,'NATIONAL','random'); k2 = get_task(df,'NATIONAL','temporal')
ytr, yte = k.y_train.astype(int), k.y_test.astype(int)
def run(cols, kk=k, label=''):
    Xtr, Xte = kk.X_train[cols], kk.X_test[cols]
    m = HGB(max_depth=6, learning_rate=0.1, max_iter=200, early_stopping=False, class_weight='balanced', random_state=0).fit(Xtr, kk.y_train.astype(int))
    p = m.predict(Xte); yt = kk.y_test.astype(int)
    return accuracy_score(yt,p), f1_score(yt,p,average='macro')
allc = list(k.X_train.columns)
print('features:', allc)
out=[]
out.append(('all baseline features',)+run(allc))
for c in ['PORTDEP','EMBPORT','ARRPORT','PLAC1TRA','SLA1PORT','MAJBUYPT','MAJSELPT','PORTRET','PLACCONS','YEARDEP','RIG']:
    if c in allc: out.append((f'{c} only',)+run([c]))
ports = [c for c in allc if c in ['PORTDEP','EMBPORT','ARRPORT','PLAC1TRA','SLA1PORT','MAJBUYPT','MAJSELPT','PORTRET','PLACCONS','CONSTREG','REGARR','REGEM1','REGDIS1','RETRNREG','EMBREG']]
out.append(('all port/place/region columns',)+run(ports))
out.append(('everything EXCEPT port/place/region',)+run([c for c in allc if c not in ports]))
out.append(('PORTDEP + PLACCONS only',)+run([c for c in ['PORTDEP','PLACCONS'] if c in allc]))
# departure port alone: how pure is the mapping port->country
d = df.loc[k.X_train.index.union(k.X_test.index)]
g = d.dropna(subset=['PORTDEP','NATIONAL']).groupby('PORTDEP').NATIONAL.agg(lambda s: s.value_counts(normalize=True).iloc[0])
w = d.dropna(subset=['PORTDEP','NATIONAL']).groupby('PORTDEP').size()
print('port->majority-country purity, weighted by voyages: %.3f' % ((g*w).sum()/w.sum()))
print('share of voyages with PORTDEP missing:', round(df.PORTDEP.isna().mean(),3))
r = pd.DataFrame(out, columns=['features used','accuracy','macro_f1']).round(3); print(r.to_string(index=False))
r.to_csv('nationality_signal_check.csv', index=False)
# temporal for contrast
r2=[]
for nm,cols in [('all',allc),('PORTDEP only',['PORTDEP']),('no port/place',[c for c in allc if c not in ports])]:
    r2.append((nm,)+run(cols,k2))
print('TEMPORAL split'); print(pd.DataFrame(r2,columns=['features','accuracy','macro_f1']).round(3).to_string(index=False))
