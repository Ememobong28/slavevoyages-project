# SlaveVoyages project: filling in missing values, and measuring how well it works

The SlaveVoyages database records about 36,000 trans-Atlantic slave voyages between 1514 and 1866. Many records have gaps: a ship's nationality, its tonnage, where it left from. The database's authors filled some of those gaps with their own estimates ("imputed" values), but nobody has measured how good a computer-made guess would be, or which gaps can be filled reliably at all.

This project asks that question in a way that can be checked:

> **Hide values we actually know, let a model guess them, and score the guesses.**

If a model can recover known values well, its guesses for unknown ones are worth taking seriously. If it can't, we learn which columns not to trust, which is just as useful to a historian.

This is research in progress. The code, the database and every result in this repository are meant to be read, questioned and re-run.

---

## The short version: what we have found so far

| Question | Answer so far | Details |
| --- | --- | --- |
| Can a model recover a ship's **nationality**? | Yes. About 96% right on hidden values. | [step 2](results/step2/STEP2_FEATURE_IMPORTANCE.md), [analysis](analysis/NATIONALITY_FINDINGS.md) |
| Why is that so easy? | The port a ship left from nearly gives the country away (Liverpool is British, Lisbon is Portuguese). The year and the ship's rig also help. | same |
| Can a model recover a ship's **tonnage**? | Partly. Typical miss is about 17% on a random test, 14% with TabPFN. Poor for very small and very large ships. | [step 1](results/STEP1_TONNAGE_RELATIVE_ERROR.md), [step 3](results/step3/STEP3_WHY_TONNAGE_IS_HARD.md) |
| Does it still work on **later voyages** when trained on earlier ones? | No. For tonnage, every model does worse than guessing the median. Ships change over time, and many records are thin. | step 3 |
| Is the dataset's imputed nationality a copy of the recorded one? | Yes, wherever both exist (26,541 of 26,543 match). It adds information only for 7,377 voyages with no recorded nationality. | [step 4](results/step4/STEP4_IMPUTED_NATIONALITY.md) |
| Does our model agree with those imputed values? | 93% overall, 97% where the model is confident. Neither side is ground truth, so this is agreement, not accuracy. | step 4 |

A fair warning: results come from a random split with fixed settings. Please read the limits in [DESIGN.md](DESIGN.md) before quoting numbers.

---

## Start here

**If you are a historian or work with the data (no coding needed)**
1. Read the table above, then the plain-language notes behind it (links in the table). Each has a short "what it means" section.
2. To browse the data yourself, build `voyages.sqlite` (see "Try it" below; it needs the original CSV) and open it in [DB Browser for SQLite](https://sqlitebrowser.org/) and look at the view `voyages_imputed_readable`. It shows names (Great Britain, Rio de Janeiro, Brig) instead of codes.
3. Questions for you are collected in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), for example how the imputed nationality was produced, and what "tons" means across centuries.

**If you are a researcher who wants to reproduce or extend the work**
1. Build the database and run the checks (below, about 5 minutes).
2. Read [DESIGN.md](DESIGN.md) (why the test is built the way it is), then `evaluate.py` and `models.py`.
3. Run `python run_experiment.py ...` and look at `results/`. To try your own model, see "Add a model".

**If you want to try TabPFN, gradient boosting and the others**
Run the commands under "Try it", or open `notebooks/run_tabpfn_colab.ipynb` in Google Colab for TabPFN (it needs a free key from [Prior Labs](https://priorlabs.ai/tabpfn-3-5), and the notebook says where to put it).

---

## How the pieces fit together

```
tastdb-exp-2019.csv          the original export. Never changed. Not included here (see DATA.md).
        |
        |  voyage_db.py       cleans it, writes one SQLite file
        v
voyages.sqlite               the database: all 276 columns, plus two views
        |                       voyages_real     = only values the sources recorded
        |                       voyages_imputed  = recorded values, gaps filled with the authors' estimates
        |
        |  evaluate.py        hides known values, splits train / validation / test,
        |                     removes columns that would leak the answer
        v
models.py                    decision tree, gradient boosting, random forest, TabPFN (plug in more)
        |
        |  run_experiment.py  trains, predicts, scores
        v
results/ and analysis/       tables, figures and short written findings (steps 1-4)
```

### Words used in this project
- **Imputed value**: an estimate filled in by the database's authors where the record had no value.
- **Real mode / imputed mode**: real uses only recorded values; imputed also uses the authors' estimates. Real is the cleaner test.
- **Masked evaluation**: hide some known values, predict them, compare with the truth.
- **Leakage**: a column that gives away the answer (for example a column that is itself derived from the thing being predicted). We remove these for each target.
- **Random split / time-based split**: the random split hides voyages at random. The time-based split trains on earlier voyages and tests on later ones, the harder and more honest test for filling in later records.
- **Accuracy, macro-F1**: how often a category is right; and the same averaged over categories so rare ones count equally.
- **Percent error**: for tonnage, the miss as a share of the true value. 17 tons off means little for a 600-ton ship and a lot for a 40-ton one.
- **Feature importance**: how much a model's score drops when one column is scrambled; a measure of what the model relies on, not of cause and effect.
- **TabPFN**: a pretrained model built for tables, used here as a strong off-the-shelf comparison.

---

## What is in this repository

| Path | What it is | When you need it |
| --- | --- | --- |
| `README.md` | this page | first |
| `DATA.md` | where to get the data and how to put it here | before running anything |
| `QUERIES.md` | worked SQL examples | when exploring the database |
| `DESIGN.md` | every design decision, with the reason | before trusting a number |
| `OPEN_QUESTIONS.md` | what we are unsure about, and who could answer | to help us |
| `voyage_db.py` | builds the database; defines real vs imputed modes | to rebuild or change the data |
| `evaluate.py` | splits, leakage rules, integrity checks | to change the test |
| `models.py` | the model registry | to add or change a model |
| `run_experiment.py` | runs and compares models | to reproduce results |
| `tests/` | 10 checks the framework must keep passing | after any change |
| `notebooks/` | the Colab notebook for TabPFN | to run TabPFN |
| `column_audit_276.csv`, `codebook_maps.json` | what we decided for each of the 276 columns; code-to-name labels from the codebook | to see why a column is used or left out |
| `analysis/` | scripts and notes behind the findings | to see how a finding was produced |
| `results/` | the outputs, one folder per step | to see the numbers |

### The results, in the order they were produced
| Step | Question | Where |
| --- | --- | --- |
| Training-share sweep | How does accuracy change with more or less training data? | `results/train_share_sweep.csv`, `analysis/train_share_sweep.py` |
| Nationality | Why is it so easy? | `analysis/NATIONALITY_FINDINGS.md` |
| 1. Tonnage percent error | How big is the typical tonnage miss, as a percent? | `results/STEP1_TONNAGE_RELATIVE_ERROR.md` |
| 2. Feature importance | Which columns drive nationality and tonnage? | `results/step2/` |
| 3. Why tonnage is hard | Where is the error (ship size, period, missing records)? | `results/step3/` |
| 4. Imputed vs model | Is the imputed nationality a copy, and does the model agree? | `results/step4/` |

Each folder has a written note (`STEP*.md`), the numbers as CSV, and figures.

---

## Try it (5 minutes)
```bash
git clone https://github.com/Ememobong28/slavevoyages-project && cd slavevoyages-project
pip install -r requirements.txt
# put tastdb-exp-2019.csv in this folder (see DATA.md)
python voyage_db.py sqlite tastdb-exp-2019.csv voyages.sqlite     # build the database (about 20 MB)
python tests/test_framework.py                                    # 10 checks
python run_experiment.py --csv voyages.sqlite --target NATIONAL_GROUP --mode real --train-frac 0.5
```
Targets so far: `NATIONAL` (20 nationalities), `NATIONAL_GROUP` (the 8 coarser groups the authors use) and `TONNAGE`. Useful options: `--mode real|imputed`, `--scheme random|temporal`, `--train-frac 0.7`, `--seeds 5`, `--models tree,gradient_boosting,random_forest`.

To repeat the written analyses:
```bash
python analysis/step2_feature_importance.py --csv voyages.sqlite --out results/step2
python analysis/step3_tonnage_why.py        --csv voyages.sqlite --out results/step3
python analysis/step4_imputed_comparison.py --csv voyages.sqlite --out results/step4
```

### Try TabPFN
Open `notebooks/run_tabpfn_colab.ipynb` in Google Colab (GPU runtime). Locally: `pip install -r requirements-tabpfn.txt`, set `TABPFN_TOKEN`, then add `tabpfn` to `--models`.

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
Any SQLite tool works (DB Browser for SQLite, DBeaver, the `sqlite3` command, R, DuckDB). More examples are in `QUERIES.md`.

## What is in the database
| Object | What it is |
| --- | --- |
| `voyages` | All 276 original columns, cleaned. Missing is `NULL`. |
| `voyages_real` (view) | Only values the sources recorded. Imputed columns are `NULL`. |
| `voyages_imputed` (view) | Recorded value where there is one, otherwise the imputed value. |
| `voyages_real_readable`, `voyages_imputed_readable` (views) | The same two tables with coded columns shown as names. A code the codebook does not define shows as `(undefined code N)`. |
| `imputation_map` | Which imputed column fills which recorded column, with a confidence note. |
| `column_audit` | Keep / reshape / defer / exclude for all 276 columns, with the reason. |
| `category_labels`, `place_labels` | Code to name tables from the codebook. |

Missing values are `NULL` in the database. `python voyage_db.py flat` writes flat tables where they read `UNKNOWN`.

## Add a model
```python
from models import register

@register('my_model', kinds=('classification',))
def my_model(task):
    # learn from task.X_train, task.y_train; tune on task.X_val, task.y_val
    return predictions_for(task.X_test)
```
Then `python run_experiment.py --csv voyages.sqlite --models tree,my_model`. A graph-based model fits the same way.

## Limits, in plain words
- Predictors are the columns the audit marks as usable; for each target we remove the columns that would give the answer away.
- Most results use one fixed set of settings and a small number of random splits. Time-based results come from a single, fixed split.
- Feature importance shows what a model leans on, not what caused what.
- Agreement with the dataset's imputed values is not accuracy; we do not know how those values were made.
- Only nationality and tonnage have been studied so far. Predicting every column and comparing with large language models are planned next.

## Data and license
The data is not included; see `DATA.md` for where to get it and its terms. Code license: to be decided. Until a license is added, all rights are reserved and the code may not be reused.
