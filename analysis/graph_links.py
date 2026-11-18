"""
Which links between voyages are informative?  (groundwork for GraphPFN)
For each candidate link rule we ask, among linked pairs where both values are recorded:
  - how often do the two voyages share a nationality group?          (compare: random pairs)
  - how far apart are their tonnages (typical % difference)?          (compare: random pairs)
and how many voyages the rule connects, in particular voyages with a MISSING value that touch a neighbour with a KNOWN value.
Recorded values only. Run: python analysis/graph_links.py --csv voyages.sqlite --out results/graph
"""
import argparse, os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import voyage_db

CAP = 40          # groups larger than this are sampled, not fully paired
def pairs_for(keys, year, window=None, rng=None):
    out = []
    for _, idx in pd.Series(np.arange(len(keys))).groupby(keys.values):
        idx = idx.values
        if len(idx) < 2: continue
        if len(idx) <= CAP: pr = list(itertools.combinations(idx, 2))
        else:
            a = rng.choice(idx, CAP * 5); b = rng.choice(idx, CAP * 5); pr = [(x, y) for x, y in zip(a, b) if x != y]
        out += pr
    P = np.array(out) if out else np.zeros((0, 2), int)
    if window is not None and len(P):
        yy = year.values; ok = np.abs(yy[P[:, 0]] - yy[P[:, 1]]) <= window; P = P[ok]
    return P

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default='voyages.sqlite'); ap.add_argument('--out', default='results/graph'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    base = voyage_db.read_base(a.csv); real = voyage_db.build_view(base, 'real')
    nat = real['NATIONAL_GROUP']; ton = real['TONNAGE']; year = base['YEARDEP'].fillna(-1); n = len(base)
    s = lambda c: base[c].astype('string').str.strip().str.upper()
    ship, capt, own, port = s('SHIPNAME'), s('CAPTAINA'), s('OWNERA'), real['PORTDEP'].astype('string')
    rules = {
      'same ship name': (ship, None), 'same ship name, within 10 years': (ship, 10),
      'same ship name, within 30 years': (ship, 30),
      'same captain (first captain)': (capt, None), 'same captain, within 10 years': (capt, 10),
      'same owner (first owner)': (own, None), 'same owner, within 10 years': (own, 10),
      'same departure port': (port, None), 'same departure port, within 5 years': (port, 5),
    }
    # random baseline
    r = rng.integers(0, n, (200000, 2)); r = r[r[:, 0] != r[:, 1]]
    def stats(P):
        i, j = P[:, 0], P[:, 1]; res = {}
        k = nat.iloc[i].notna().values & nat.iloc[j].notna().values
        res['nat_pairs'] = int(k.sum()); res['same_nationality_pct'] = 100 * float((nat.iloc[i][k].values == nat.iloc[j][k].values).mean()) if k.any() else np.nan
        t = ton.iloc[i].notna().values & ton.iloc[j].notna().values
        res['ton_pairs'] = int(t.sum())
        res['median_ton_diff_pct'] = 100 * float(np.median(np.abs(ton.iloc[i][t].values - ton.iloc[j][t].values) / ton.iloc[j][t].values)) if t.any() else np.nan
        return res
    rows = [{'rule': 'random pair (baseline)', 'pairs': len(r), **stats(r)}]
    for name, (key, win) in rules.items():
        P = pairs_for(key.fillna('<NA>%d' % 0) if False else key.where(key.notna(), pd.Series(['__na__%d' % k for k in range(n)], index=key.index)), year, win, rng)
        deg = np.zeros(n, int); np.add.at(deg, P[:, 0], 1); np.add.at(deg, P[:, 1], 1)
        # coverage for missing targets: voyages with missing value that have >=1 neighbour with a known value
        nb_known_nat = np.zeros(n, bool); nb_known_ton = np.zeros(n, bool)
        for tgt, arr in ((nat, nb_known_nat), (ton, nb_known_ton)):
            kn = tgt.notna().values; arr[P[kn[P[:, 1]], 0]] = True; arr[P[kn[P[:, 0]], 1]] = True
        miss_n, miss_t = nat.isna().values, ton.isna().values
        rows.append({'rule': name, 'pairs': len(P), 'voyages_with_a_link_pct': round(100 * (deg > 0).mean(), 1), **stats(P),
                     'missing_nat_with_known_neighbour_pct': round(100 * (nb_known_nat & miss_n).sum() / miss_n.sum(), 1),
                     'missing_ton_with_known_neighbour_pct': round(100 * (nb_known_ton & miss_t).sum() / miss_t.sum(), 1)})
    T = pd.DataFrame(rows).round(1); T.to_csv(f'{a.out}/link_rules.csv', index=False); print(T.to_string())
if __name__ == '__main__': main()
