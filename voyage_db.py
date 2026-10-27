"""
voyage_db.py  --  clean SlaveVoyages database with a real/imputed switch.

What this does
  1. Reads the ORIGINAL csv (never modified).
  2. Turns every kind of empty ('' , whitespace, ',,') into ONE missing value.
  3. Builds a "real" view (only values the sources recorded) and an "imputed" view
     (recorded value where there is one, otherwise the authors' imputed value).
  4. build_sqlite() writes ONE SQLite database with the two modes as views; export_flat() writes UNKNOWN tables.

The pairs below (which imputed column replaces which recorded column) are MY READING of the
codebook (section 3). They are data, not code, so they are easy to change: edit IMPUTATION_MAP.
"""
import sqlite3
import numpy as np, pandas as pd

UNKNOWN = 'UNKNOWN'          # the one symbol used for a missing value in exported tables
MODES = ('real', 'imputed')

# concept column -> (recorded column, imputed column, how_sure, note)
IMPUTATION_MAP = {
 'PORTDEP':  ('PORTDEP',  'PTDEPIMP',  'high', 'Same code space; the two agree on 99.99% of voyages where both exist.'),
 'MAJBUYPT': ('MAJBUYPT', 'MJBYPTIMP', 'high', 'Same geographic code space.'),
 'MAJSELPT': ('MAJSELPT', 'MJSLPTIMP', 'high', 'Same geographic code space.'),
 'YEARDEP':  ('DATEDEPC', 'YEARDEP',   'medium', 'Recorded year voyage began vs imputed year. YEARDEP has no missing values.'),
 'YEARAF':   ('DLSLATRC', 'YEARAF',    'medium', 'Recorded year left last slaving port vs imputed year departed Africa.'),
 'YEARAM':   ('DATARR34', 'YEARAM',    'medium', 'Recorded year of first disembarkation vs imputed year of arrival.'),
 'SLAXIMP':  ('TSLAVESD', 'SLAXIMP',   'low', 'Recorded slaves on board at departure is only an approximate counterpart of imputed total embarked.'),
 'SLAMIMP':  ('SLAARRIV', 'SLAMIMP',   'low', 'Recorded slaves arrived at first port is only an approximate counterpart of imputed total disembarked.'),
}
# NATIONAL needs a crosswalk: NATINIMP uses coarser groups than NATIONAL.
NATIONAL_TO_GROUP = {1:3, 2:3, 3:3, 4:6, 5:6, 6:6, 7:7, 8:8, 9:9, 10:10,
                     11:15, 12:15, 13:15, 14:15, 16:30, 17:30, 18:30, 19:30, 24:30, 25:30}
# imputed-only columns with no recorded twin; present in 'imputed' mode, UNKNOWN in 'real' mode
IMPUTED_ONLY = ['MAJBYIMP', 'MAJBYIMP1', 'MJSELIMP', 'MJSELIMP1', 'TONMOD']


def read_clean(csv):
    """Original csv -> DataFrame where every empty form is NaN. Nothing is written back."""
    df = pd.read_csv(csv, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == 'str':
            s = df[c].astype('string').str.strip()
            df[c] = s.mask(s.isin(['', ',,']))
    return df


def build_view(df, mode):
    """Return the table for one mode. Same columns in both modes; only the values differ."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    out = df.copy()
    # nationality in coarse groups, so the two modes use the same vocabulary
    group = df['NATIONAL'].map(NATIONAL_TO_GROUP)
    out['NATIONAL_GROUP'] = group
    if mode == 'imputed':
        out['NATIONAL_GROUP'] = group.fillna(df['NATINIMP'])
    for concept, (rec, imp, _, _) in IMPUTATION_MAP.items():
        if mode == 'real':
            out[concept] = df[rec]
        else:
            out[concept] = df[rec].where(df[rec].notna(), df[imp])
    if mode == 'real':
        for c in IMPUTED_ONLY + ['NATINIMP', 'PTDEPIMP', 'MJBYPTIMP', 'MJSLPTIMP']:
            if c in out:
                out[c] = np.nan
    return out


def provenance(df, mode):
    """Per concept column: how many cells are recorded / imputed / unknown in this mode."""
    rows = []
    for concept, (rec, imp, sure, note) in IMPUTATION_MAP.items():
        rec_known = df[rec].notna(); imp_known = df[imp].notna()
        n_rec = int(rec_known.sum()); n_imp = int((~rec_known & imp_known).sum()) if mode == 'imputed' else 0
        rows.append(dict(column=concept, recorded=n_rec, filled_from_imputed=n_imp,
                         unknown=int(len(df) - n_rec - n_imp), recorded_column=rec, imputed_column=imp,
                         confidence=sure, note=note))
    g = df['NATIONAL'].map(NATIONAL_TO_GROUP).notna(); i = df['NATINIMP'].notna()
    n_imp = int((~g & i).sum()) if mode == 'imputed' else 0
    rows.append(dict(column='NATIONAL_GROUP', recorded=int(g.sum()), filled_from_imputed=n_imp,
                     unknown=int(len(df) - g.sum() - n_imp), recorded_column='NATIONAL', imputed_column='NATINIMP',
                     confidence='high', note='Recorded NATIONAL is mapped to the coarser NATINIMP groups (crosswalk in voyage_db.py).'))
    return pd.DataFrame(rows)


def load_database(csv='tastdb-exp-2019.csv', mode='real', marker=None, drop_empty=True, df=None):
    """
    The one function to call.
      mode   : 'real'    -> only recorded values
               'imputed' -> recorded values, with imputed values filling the gaps
      marker : None      -> missing stays NaN (what models want)
               'UNKNOWN' -> missing is the text UNKNOWN (what a person wants to read)
      drop_empty : drop columns that are 100% missing in this mode
    """
    base = df if df is not None else read_base(csv)
    v = build_view(base, mode)
    if drop_empty:
        v = v.loc[:, v.notna().any()]
    if marker is not None:
        v = v.astype('object').where(v.notna(), marker)
    return v


def _sql_views(columns):
    """CREATE VIEW statements for the two modes, generated from IMPUTATION_MAP so SQL and Python cannot drift."""
    concept = set(IMPUTATION_MAP); twins = {v[1] for v in IMPUTATION_MAP.values()} - concept
    blank_in_real = set(IMPUTED_ONLY) | twins | {'NATINIMP'}
    cw = ' '.join(f'WHEN {k} THEN {v}' for k, v in NATIONAL_TO_GROUP.items())
    group = f'CASE CAST(NATIONAL AS INTEGER) {cw} END'
    q = lambda c: f'"{c}"'
    def select(mode):
        parts = []
        for c in columns:
            if c in IMPUTATION_MAP:
                rec, imp = IMPUTATION_MAP[c][:2]
                parts.append(f'{q(rec)} AS {q(c)}' if mode == 'real' else f'COALESCE({q(rec)}, {q(imp)}) AS {q(c)}')
            elif c in blank_in_real and mode == 'real':
                parts.append(f'NULL AS {q(c)}')
            else:
                parts.append(q(c))
        parts.append(f'{group} AS NATIONAL_GROUP' if mode == 'real' else f'COALESCE({group}, "NATINIMP") AS NATIONAL_GROUP')
        return ',\n  '.join(parts)
    return {m: f'CREATE VIEW voyages_{m} AS SELECT\n  {select(m)}\nFROM voyages;' for m in MODES}


def build_sqlite(csv='tastdb-exp-2019.csv', path='voyages.sqlite', audit='column_audit_276.csv', maps='codebook_maps.json'):
    """
    Build ONE SQLite file you can open in any SQL tool (DB Browser, DBeaver, sqlite3, pandas, R, DuckDB).
      voyages           every original column, cleaned; missing = NULL (SQL's own "unknown"); original CSV untouched
      voyages_real      VIEW: recorded values only
      voyages_imputed   VIEW: recorded values, gaps filled with imputed values
      imputation_map    which imputed column fills which recorded column, and how sure the pairing is
      column_audit      the keep/reshape/defer/exclude decision and reason for each of the 276 columns
      category_labels   code -> label for the coded columns (nationality, rig, fate, ...)
      place_labels      code -> label for the 5-digit port/place codes
    Missing values: NULL inside the database; the flat exports print them as UNKNOWN (see `export_flat`).
    """
    import json, os
    base = read_clean(csv)
    if os.path.exists(path): os.remove(path)
    con = sqlite3.connect(path)
    base.to_sql('voyages', con, index=False)
    con.execute('CREATE UNIQUE INDEX idx_voyageid ON voyages(VOYAGEID)')
    for c in ('NATIONAL', 'PORTDEP', 'YEARDEP'): con.execute(f'CREATE INDEX idx_{c} ON voyages({c})')
    for m, sql in _sql_views(list(base.columns)).items(): con.execute(sql)
    pd.DataFrame([dict(concept=k, recorded_column=v[0], imputed_column=v[1], confidence=v[2], note=v[3]) for k, v in IMPUTATION_MAP.items()]
                 ).to_sql('imputation_map', con, index=False)
    if os.path.exists(audit):
        pd.read_csv(audit).to_sql('column_audit', con, index=False)
    if os.path.exists(maps):
        M = json.load(open(maps))
        pd.DataFrame([(c, int(k), v) for c, d in M['categorical'].items() for k, v in d.items()],
                     columns=['column_name', 'code', 'label']).to_sql('category_labels', con, index=False)
        pd.DataFrame([(int(k), v) for k, v in M['geographic'].items()], columns=['code', 'label']).to_sql('place_labels', con, index=False)
    con.commit(); con.close()
    return path


def read_base(source='tastdb-exp-2019.csv'):
    """The cleaned table, from either the original csv or a .sqlite file made by build_sqlite()."""
    if str(source).endswith(('.sqlite', '.db')):
        con = sqlite3.connect(source); df = pd.read_sql('SELECT * FROM voyages', con); con.close(); return df
    return read_clean(source)


def read_view(path, mode):
    """The ready-made view from the sqlite file (for people who prefer to start from SQL)."""
    con = sqlite3.connect(path); df = pd.read_sql(f'SELECT * FROM voyages_{mode}', con); con.close(); return df


def export_flat(csv='tastdb-exp-2019.csv', outdir='.'):
    """Flat tables for both modes with every missing cell written as UNKNOWN (for reading, or for tools that want one big table)."""
    base = read_base(csv); report = {}
    for mode in MODES:
        flat = load_database(mode=mode, marker=UNKNOWN, df=base)
        flat.to_csv(f'{outdir}/voyages_flat_{mode}.csv', index=False)
        report[mode] = dict(rows=len(flat), columns=flat.shape[1])
    return report


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'sqlite'
    if cmd == 'sqlite': print(build_sqlite(*sys.argv[2:]))
    elif cmd == 'flat': print(export_flat(*sys.argv[2:]))
    else: raise SystemExit('usage: python voyage_db.py [sqlite|flat] [csv] ...')
