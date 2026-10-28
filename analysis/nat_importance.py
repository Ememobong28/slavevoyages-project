import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import accuracy_score
from eval_pipeline import *
df = load_clean(); k = get_task(df,'NATIONAL','random')
ports = ['PORTDEP','EMBPORT','ARRPORT','PLAC1TRA','SLA1PORT','MAJBUYPT','MAJSELPT','PORTRET','PLACCONS','CONSTREG','REGARR','REGEM1','REGDIS1','RETRNREG','EMBREG']
cols=[c for c in k.X_train.columns if c not in ports]
mk=lambda: HGB(max_depth=6,learning_rate=0.1,max_iter=200,early_stopping=False,class_weight='balanced',random_state=0)
m=mk().fit(k.X_train[cols],k.y_train.astype(int)); yt=k.y_test.astype(int)
base=accuracy_score(yt,m.predict(k.X_test[cols])); rng=np.random.default_rng(0); rows=[]
for c in cols:
    X=k.X_test[cols].copy(); X[c]=rng.permutation(X[c].values); rows.append((c,base-accuracy_score(yt,m.predict(X))))
r=pd.DataFrame(rows,columns=['column','accuracy_drop_when_shuffled']).sort_values('accuracy_drop_when_shuffled',ascending=False).round(3)
print('base',round(base,3)); print(r.head(10).to_string(index=False))
# missingness-only: does WHICH columns are filled in predict nationality?
M=k.X_train[cols].isna().astype(int); Mt=k.X_test[cols].isna().astype(int)
mm=mk().fit(M,k.y_train.astype(int)); print('missingness pattern only (no actual values): acc', round(accuracy_score(yt,mm.predict(Mt)),3))
r.to_csv('nationality_importance_noports.csv',index=False)
