# Which columns drive the predictions? (step 2)

Setup: recorded values only, random split, 70% of known values for training, gradient boosting. For each column we shuffle it in the test set and measure how much the score gets worse (10 shuffles x 3 splits). Columns that carry overlapping information are also shuffled in groups, and a model is retrained without each group. Numbers are in the CSV files next to this note.

## Nationality (accuracy 96.0%, macro-F1 62.8%)

| Column group | Accuracy lost when shuffled | Accuracy when group removed and retrained |
| --- | --- | --- |
| Departure and purchase ports/regions | 32.7 points | 93.9% |
| Year of the voyage | 13.4 points | 94.5% |
| Landing and return ports/regions | 7.1 points | 95.1% |
| Everything else | under 1 point each | 95.6-96.0% |

- One column, **PORTDEP (port of departure), accounts for 27.6 points** on its own, and the year (YEAR5) for 8.9.
- Why: a port tells you the country. A ship leaving Liverpool is British, one leaving Lisbon is Portuguese. This is close to a lookup, not a deep pattern.
- Removing a whole group and retraining costs only 1-2 points, because the other port and year columns cover for it. Shuffling overstates how much any one group is needed; retraining shows how much is left.
- The high accuracy is mostly the large nationalities. Macro-F1 (62.8%) is much lower, so small nationalities are predicted poorly (see `confusion_NATIONAL.png`).

## Tonnage (typical error 17.0%, 64.5% within 25%)

| Column group | Typical error gets worse by | Typical error when group removed and retrained |
| --- | --- | --- |
| Captives (intended, embarked, landed) | 14.5 points | 21.7% |
| Ship size and type (rig, type of tonnage, guns) | 9.8 points | 17.9% |
| Year of the voyage | 9.3 points | 19.0% |
| Departure ports/regions | 3.7 points | 17.3% |
| Everything else | under 2 points | about 17% |

- No single column dominates, unlike nationality. The strongest are the year (6.4), the tonnage type (6.1), intended captives (5.3), captives landed (4.9) and rig (3.8).
- The best predictors are mostly **missing a lot**: intended captives 66% missing, captives embarked 76%, guns 67%, rig 38%. The model has little to work with for many voyages.
- Retraining without the captives group raises the error from 17.0% to 21.7%. The captive columns carry real information: bigger ships carry more people.
- Nothing here says why tonnage is hard. That is step 3.

## Agreement check
The decision tree's own importance ranks tonnage columns much like boosting does (rank agreement 0.92) and nationality columns less so (0.59), because many port columns overlap and the tree splits credit among them. For nationality, trust the shuffling results and the retraining table over the tree's ranking.

## Caveats
- Importance describes what this model relies on, not what causes what.
- Shuffled-column importance is unreliable when columns are strongly related (ports and regions); that is why the group and retraining tables are there.
- One model (gradient boosting), one training share (70%).
