# SlaveVoyages project

A small framework for asking one question fairly: **how well can a model fill in missing values in the SlaveVoyages 2019 database?**

It does three things:

1. **A real database you can query.** One SQLite file with the cleaned data, a `voyages_real` view (recorded values only) and a `voyages_imputed` view (recorded values, gaps filled with the authors' imputed values), plus the codebook labels and the column-by-column audit.
2. **A fair test.** Hide values we know, let a model guess them, score the guesses. The training share is a parameter. Leakage columns are removed per target. A time-based split is available as a stress test.
3. **Models you can swap.** Decision tree, gradient boosting, random forest and [TabPFN](https://priorlabs.ai/tabpfn-3-5) are included. Adding another is one decorated function.

> Status: research work in progress. One random seed per run, three targets so far (nationality, tonnage). Results are in `results/` and `analysis/`; limits are in `DESIGN.md`.

## Try it (5 minutes)
```bash
git clone https://github.com/Ememobong28/slavevoyages-project && cd slavevoyages-project
pip install -r requirements.txt
# put tastdb-exp-2019.csv in this folder (see DATA.md)
python voyage_db.py sqlite tastdb-exp-2019.csv voyages.sqlite     # build the database (about 20 MB)
python tests/test_framework.py                                    # 9 checks
python run_experiment.py --csv voyages.sqlite --target NATIONAL_GROUP --mode real --train-frac 0.5
```

### Try TabPFN
Open `notebooks/run_tabpfn_colab.ipynb` in Google Colab (GPU runtime). TabPFN needs a free key from Prior Labs; the notebook says where to put it. Locally: `pip install -r requirements-tabpfn.txt`, set `TABPFN_TOKEN`, then add `tabpfn` to `--models`.

### Query the database
```python
import sqlite3, pandas as pd
con = sqlite3.connect('voyages.sqlite')
print(pd.read_sql("""
  SELECT l.label AS nationality, COUNT(*) AS voyages
  FROM voyages_real v JOIN category_labels l ON l.column_name = 'NATIONAL' AND l.code = v.NATIONAL
  GROUP BY 1 ORDER BY 2 DESC LIMIT 5""", con))
# or let the readable view do the join for you:
print(pd.read_sql("SELECT NATIONAL, PORTDEP, RIG FROM voyages_imputed_readable LIMIT 5", con))
```
More worked examples, including switching between real and imputed, are in `QUERIES.md`. Any SQLite tool works (DB Browser for SQLite, DBeaver, the `sqlite3` command, R, DuckDB).

## What is in the database
| Object | What it is |
| --- | --- |
| `voyages` | All 276 original columns, cleaned. Missing is `NULL`. |
| `voyages_real` (view) | Only values the sources recorded. Imputed columns are `NULL`. |
| `voyages_imputed` (view) | Recorded value where there is one, otherwise the imputed value. |
| `voyages_real_readable`, `voyages_imputed_readable` (views) | The same two tables with coded columns shown as names (Great Britain, Rio de Janeiro, Brig, ...). A code the codebook does not define shows as `(undefined code N)`. |
| `imputation_map` | Which imputed column fills which recorded column, with a confidence note. |
| `column_audit` | Keep / reshape / defer / exclude for all 276 columns, with the reason. |
| `category_labels`, `place_labels` | Code to label tables from the codebook. |

Missing values are `NULL` in the database (SQL's own "unknown"). `python voyage_db.py flat` writes flat tables where they read `UNKNOWN`.

## Add a model
```python
from models import register

@register('my_model', kinds=('classification',))
def my_model(task):
    # learn from task.X_train, task.y_train; tune on task.X_val, task.y_val
    return predictions_for(task.X_test)
```
`python run_experiment.py --csv voyages.sqlite --models tree,my_model`. A graph-based model fits the same way.

## Layout
| Path | |
| --- | --- |
| `voyage_db.py` | build the database; load either mode |
| `evaluate.py` | splits (with `train_frac`), leakage drop lists, integrity checks |
| `models.py` | the model registry |
| `run_experiment.py` | run and compare models |
| `tests/` | rules the framework must keep |
| `DESIGN.md`, `OPEN_QUESTIONS.md` | decisions with reasons; questions still open |
| `analysis/` | why nationality is easy to predict, and the training-share sweep |
| `eval_pipeline.py`, `column_audit_276.csv`, `codebook_maps.json` | metrics, the column audit, codebook labels |

## License
To be decided. Until a license is added, all rights are reserved and the code may not be reused. The data is not included and has its own terms.
