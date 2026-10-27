# Why is nationality so easy to predict? (question from the Oct 23 meeting)

Hypothesis from the meeting: it is mostly "departure port tells you the country", so a 90%+ score is not surprising, and the real missing rate is probably near zero.

## What I found
All numbers: gradient boosting, same settings as the earlier results, random split, 20-code `NATIONAL`. The earlier 95.7% reproduces (95.9% here).

1. **Nationality is not near zero missing.** `NATIONAL` is blank for 9,538 of 36,108 voyages (26.4%). The authors' imputed column `NATINIMP` fills 7,377 of those; 2,161 voyages have neither.
2. **The port is a strong clue, not the whole story.**
   - Departure port alone gives 68% accuracy (macro-F1 0.36), against 42% for always guessing the most common country.
   - Where a departure port is known, its most common country covers 94.9% of voyages. But 19% of voyages have no departure port.
3. **Remove every port, place and region column and it is still 88%.** (All features: 96%.) What carries it, measured by how much accuracy drops when a column is shuffled:

   | Column | Drop in accuracy |
   | --- | --- |
   | `RIG` (rig of the vessel) | 18.5 points |
   | `YEARDEP` (year) | 18.0 |
   | `TONNAGE` | 12.5 |
   | `YEAR5` | 7.2 |
   | `SLINTEND` (slaves intended) | 5.6 |
   | `GUNS` | 3.7 |

   My guess, not tested: nations used different ship types and sizes in different eras, so rig + year + tonnage points to a nation.
4. **It is not the same ship appearing in training and test.** 80% of random test voyages share a ship name with a training voyage, so I held out whole ships (grouped by `SHIPNAME`). Accuracy stayed at 96.5%. (Matching on name is imperfect: two ships can share a name.)
5. **Which columns are filled in tells you a little.** A model that sees only the blank/not-blank pattern, with no values, scores 51%.
6. **It needs very little training data.** With only 10% of known values for training (about 2,650 voyages) accuracy is already 95.5% (grouped nationality, table below).

## The caution that matters most
The voyages whose nationality is actually missing look different from the ones we test on:

| Column | Missing when nationality is known | Missing when nationality is missing |
| --- | --- | --- |
| `TONNAGE` | 38% | 88% |
| `YRCONS` (year built) | 77% | 99.5% |
| `GUNS` | 77% | 98% |
| `PORTDEP` | 14% | 35% |

So the score on known voyages probably **overstates** what we could do for the truly missing ones: they have fewer clues. The 51% from the time-based split (and 38% for gradient boosting) is a better hint at how hard it gets when the training data does not look like the target data.

## Training share (grouped nationality, 8 groups; accuracy on test)
| Train share | Real mode: gradient boosting | Real: tree | Imputed mode: gradient boosting | Imputed: tree |
| --- | --- | --- | --- | --- |
| 10% | 0.955 | 0.928 | 0.954 | 0.925 |
| 30% | 0.961 | 0.947 | 0.961 | 0.942 |
| 50% | 0.962 | 0.942 | 0.966 | 0.950 |
| 70% | 0.965 | 0.944 | 0.962 | 0.949 |

Always guessing the most common group: 42% (real), 35% (imputed). Test sets differ in size between rows (they are what is left after training), so compare patterns, not decimals.
In imputed mode the "answers" include the authors' imputed values, so that score measures agreement with their imputation.

## Tonnage (real mode; average miss in tons, lower is better)
| Train share | Median guess | Tree | Gradient boosting |
| --- | --- | --- | --- |
| 10% | 75.3 | 53.5 | 43.9 |
| 30% | 74.0 | 46.2 | 38.3 |
| 50% | 75.1 | 44.3 | 36.9 |
| 70% | 72.2 | 43.2 | 35.5 |

Unlike nationality, tonnage keeps improving with more training data.

## Not done
One seed throughout. The importance numbers come from shuffling one column at a time, so correlated columns share credit. No check yet whether the rig/year/tonnage rule matches what historians would expect.
