import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from eval_pipeline import *
df = load_clean(); k = get_task(df,'NATIONAL','random')
allc = list(k.X_train.columns)
ports = ['PORTDEP','EMBPORT','ARRPORT','PLAC1TRA','SLA1PORT','MAJBUYPT','MAJSELPT','PORTRET','PLACCONS','CONSTREG','REGARR','REGEM1','REGDIS1','RETRNREG','EMBREG']
ship = df.SHIPNAME.str.lower().str.strip()
# how much of the original random test set shares a ship name with training rows?
tr_names = set(ship[k.X_train.index].dropna()); te = ship[k.X_test.index]
print('random test voyages whose ship name also appears in training: %.1f%% (of those with a name)' % (100*te.dropna().isin(tr_names).mean()))
print('voyages per ship name (median / mean):', ship.value_counts().median(), round(ship.value_counts().mean(),2))
# grouped split: whole ships held out
ok = df.NATIONAL.notna() & ~df.NATIONAL.isin([3,6]) & ship.notna()
idx = df.index[ok]; X = build_features(df, SPECS['NATIONAL']).loc[idx]; y = df.NATIONAL.loc[idx].astype(int); g = ship.loc[idx]
def fit_score(cols, tr, te):
    m = HGB(max_depth=6, learning_rate=0.1, max_iter=200, early_stopping=False, class_weight='balanced', random_state=0).fit(X.loc[tr,cols], y.loc[tr])
    p = m.predict(X.loc[te,cols]); return accuracy_score(y.loc[te],p), f1_score(y.loc[te],p,average='macro')
rows=[]
for nm,(tr,te) in {'random rows (ships can repeat)': train_test_split(idx, test_size=0.15, random_state=1),
                   'whole ships held out': next(iter([ (idx[a],idx[b]) for a,b in GroupShuffleSplit(1,test_size=0.15,random_state=1).split(idx,groups=g)]))}.items():
    for fs,cols in {'all features':allc, 'no port/place/region':[c for c in allc if c not in ports], 'PORTDEP only':['PORTDEP']}.items():
        a,f = fit_score(cols,tr,te); rows.append((nm,fs,len(te),round(a,3),round(f,3)))
r = pd.DataFrame(rows,columns=['split','features','n_test','accuracy','macro_f1']); print(r.to_string(index=False)); r.to_csv('nationality_grouped_check.csv',index=False)
