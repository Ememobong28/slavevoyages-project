# Why tonnage predicts poorly (step 3)

Same setup as step 2: recorded values only, gradient boosting, 70% of known values for training, test set of 2,651 voyages. Tables are in the CSV files next to this note; figures are `error_breakdown.png` and `drift.png`.

## 1. Small and very large ships are the problem (random split)
| True size | Voyages | Typical error | Within 25% | Average over/under |
| --- | --- | --- | --- | --- |
| under 60 tons | 221 | 52.6% | 26% | predicted 70% too high |
| 60-100 | 538 | 17.9% | 62% | +19% |
| 100-150 | 841 | 15.4% | 73% | +1% |
| 150-250 | 659 | 14.2% | 70% | -8% |
| 250-400 | 319 | 13.3% | 66% | -12% |
| over 400 | 73 | 28.1% | 47% | -27% |

- The middle of the range (100-400 tons, most voyages) is predicted to within about 14-15%.
- Small ships are predicted too big and large ships too small: the model pulls everything toward the typical ship. Percent error punishes small ships hardest, because a 30-ton miss on a 40-ton ship is 75%.
- This is the main reason the overall number looks weak. The average error in tons is largest for big ships (172 tons over 400), but as a percent it is the tiny ships that fail.

## 2. Information per voyage matters a lot
Counting how many of 8 helper columns (rig, guns, tonnage type, intended / embarked / landed captives, year built, year registered) a voyage has:

| Helper columns known | Voyages | Typical error |
| --- | --- | --- |
| 0-2 | 849 | 23.9% |
| 3-4 | 927 | 17.4% |
| 5-6 | 685 | 12.1% |
| 7-8 | 190 | 11.9% |

Voyages with a lot recorded are predicted well (about 12%); about a third of test voyages have almost nothing and get about 24%. The model is limited by the records, not only by the method.

## 3. The data is thin in the places that matter
- Tonnage is missing for 33% of voyages in 1750-99 but 69% in 1800-24 and 78% before 1700.
- Guns are missing for 70-99% of voyages in every period, year built for 68-99%, and intended captives for 64-99%.
- Rig is well recorded from 1800 (8-19% missing) but not before (31-67% missing).

## 4. Why the time-based split fails
- Median tonnage moves: about 100-130 tons before 1800, 188 in 1800-24, then 134, then 190 (after 1850). The test set (the latest 30% of voyages, 1800 onward) has ships that look different from the training set.
- Trained on earlier voyages, the model over-predicts small ships by a wide margin (under-60-ton ships: predicted 286% too high on average; 60-100 tons: 125% too high), and typical error is 40-70% in every later period.
- Odd result: in the time-based split, voyages with more recorded helper columns do worse (64-67%) than ones with few (40%). I do not have an explanation. It may be that the relationships between rig, captives and size changed after 1800, so the model trusts them wrongly. This needs a closer look.
- "Tons" may not mean the same thing across the whole record (the tonnage-type column, TONTYPE, is a strong predictor, which hints at this). I have not checked how the definition changed. That is a question for the professor.

## What to take away
1. Tonnage is hard mostly because many voyages have little recorded, and because the very small and very large ships are rare.
2. A random split flatters the model; the time-based split is the honest test for imputing the later voyages, and no model beats guessing the median there.
3. Possible next checks: add an error band to each prediction so users know which voyages to trust; train on tonnage only for 1750+; ask what the tonnage types mean.

## Caveats
- One model, one split of each kind (seed 42). The time-based split is deterministic.
- Bins are chosen by hand; small bins (under 60 tons, over 400, 1850+) have few voyages.
- The explanations in section 4 are hypotheses, not tested.
