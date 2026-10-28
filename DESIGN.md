# Design decisions

Each decision has the reason next to it, so they can be argued with.

## 1. One symbol for missing
Every empty form in the original file (empty text, spaces, `,,`) becomes one missing value: `NULL` in the SQLite database, `UNKNOWN` in the flat exports.
Models receive a real missing value (NaN) so numbers stay numbers; `load_database(marker='UNKNOWN')` gives the readable form.
*Why:* the original mixes several kinds of "empty", and eight columns hid blanks as text, so they were being counted as filled (77.7% of cells are really missing, not 75.9%).

## 2. Two modes, same columns (as SQL views and in Python)
`mode='real'` uses only values the sources recorded. `mode='imputed'` keeps every recorded value and fills only the gaps with the authors' imputed value.
Both give the same column names, so the same code runs on either.
*Why:* this is the toggle asked for in the meeting. Imputed mode never overwrites a recorded value (tested).
The pairs are in `IMPUTATION_MAP` (`voyage_db.py`), as data you can edit. Confidence is recorded for each pair. Five columns are imputed-only and are empty (NULL) in real mode.

## 3. Nationality gets a crosswalk
`NATIONAL` has 20 codes. The imputed `NATINIMP` has 8 coarser groups (for example Portugal and Brazil are one group).
Filling one from the other would mix two vocabularies, so I added `NATIONAL_GROUP`: recorded `NATIONAL` mapped to the 8 groups (real mode), plus `NATINIMP` for the gaps (imputed mode).
The mapping comes from how the two columns line up in the data (it is exact apart from 2 stray rows). The old 20-class `NATIONAL` target still works for real mode.

## 4. The training share is a share of KNOWN values
`train_frac=0.5` means half of the voyages where the answer is known (in that mode), not half of all rows. The rest is split into validation and test.
*Why:* this handles the edge case raised in the meeting. Training labels can never be missing, because only known values are sampled.
If a column has too few known values, the run stops with a message that says how many training values it would have had (minimum 30). It does not run quietly on noise.
Also refused: `train_frac` outside (0, 1), a split with fewer than 2 validation/test values.
Rare classes: stratified where possible; if there are more classes than training rows the split falls back to unstratified. Test classes never seen in training are reported as a note on the task.

## 5. Validation is for tuning, test is scored once
Unchanged from before. Settings are chosen on validation only.

## 6. Leakage drops live in one place
`SPECS` in `evaluate.py`: for each target, the columns that contain or are computed from the answer. The check `check_integrity` fails the run if one is present.
`YEARDEP` is imputed but used as a predictor (and as the time axis for the time-based split). Flagged as a question.

## 7. Models are plugins
`models.py` holds a registry. A model is one function that takes the task and returns test predictions. Decorate it with `@register('name')` and it runs in the comparison.
Included: reference guess, decision tree, gradient boosting, random forest (added as a demonstration, 6 lines), TabPFN (appears only when the `tabpfn` package is installed, which needs a key and ideally a GPU).
A graph-based model would be added the same way: build whatever graph it needs from `task.X_train` and return predictions for `task.X_test`.

## 8. Rows with a truly missing target
These are kept as `task.X_missing` and never scored. They are the rows a finished method would fill in.

## What this does not do yet
- Only three targets. Place targets and dates need their own leakage lists (the region columns derive from the port columns).
- One random seed per run. Multi-seed averages are the next step.
- Missing is `NULL` inside the database (SQL's own "unknown"); `python voyage_db.py flat` prints it as `UNKNOWN` in flat files. Storing the text UNKNOWN inside numeric columns would break arithmetic and sorting.
- The SQL views and the Python code are two implementations of the same rules; a test checks they give identical tables.
- No automatic test run on GitHub yet: the tests need the data file, which cannot be committed.
