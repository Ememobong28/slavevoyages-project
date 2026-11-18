# The dataset's imputed nationality vs ours (step 4)

NATINIMP is the dataset's own imputed nationality, in 8 coarse groups. Recorded NATIONAL has 20 codes; a crosswalk (in `voyage_db.py`) maps them onto the same 8 groups.

## A. Is the imputed column just a copy of the recorded one?
| | Voyages |
| --- | --- |
| Both present | 26,543 |
| Same after the crosswalk | 26,541 |
| Different | 2 |
| Recorded only (imputed blank) | 27 |
| Imputed only (recorded blank) | 7,377 |
| Neither | 2,161 |

- **Where both exist, the imputed column is a copy** (coarser groups) of the recorded one, apart from 2 voyages (`copy_check_disagreements.csv`: voyage 1603, 1837, recorded code 4 vs imputed 15; voyage 33916, 1648, recorded 8 vs imputed 10).
- So it adds information only for the **7,377 voyages with no recorded nationality**. We can't test those values against a recorded truth, because none exists. That is why the comparison below is only agreement, not accuracy.
- Open question for the professor / the dataset's maintainers: how were those 7,377 values made (a rule, a person, or a model)?

## B. Does our model agree with them?
Gradient boosting trained on recorded nationality (as the 8 groups, 85% of known voyages for training, predictors exclude the imputed column), then predicting the 7,377 imputed-only voyages.

- **Agreement: 92.8%.** On held-out recorded voyages the same model is 96.9% accurate.
- Always guessing the most common recorded group (Great Britain) would agree with only 10.6%, because these voyages are mostly Portuguese/Brazilian (71%). "Always Portugal/Brazil" would agree 71%.

| Model's confidence | Voyages | Agree with dataset |
| --- | --- | --- |
| under 60% | 189 | 51% |
| 60-80% | 273 | 54% |
| 80-95% | 437 | 71% |
| 95%+ | 6,478 | 97% |

| Dataset's imputed group | Voyages | Agree |
| --- | --- | --- |
| Portugal/Brazil | 5,247 | 96.7% |
| Great Britain | 779 | 80.6% |
| Netherlands | 459 | 83.2% |
| U.S.A. | 477 | 91.0% |
| Spain/Uruguay | 274 | 81.8% |
| France | 116 | 72.4% |

By period: 89% before 1750, 91% in 1750-99, 97% in 1800-49, 61% after 1850 (only 56 voyages).

Biggest disagreements: dataset says Great Britain, model says U.S.A. (118 voyages); dataset Portugal/Brazil, model Spain/Uruguay (76); dataset Netherlands, model Denmark/Baltic (45).

## What it means
1. The two largely agree, and agree almost completely where the model is confident. The disagreements sit where the model is unsure, or in pairs that are hard to tell apart historically (Britain vs the U.S.A.).
2. Agreement (92.8%) is below the 96.9% on recorded voyages. Either these voyages are harder (they lack a recorded nationality, so likely have less recorded), or the dataset's imputed values are less reliable than recorded ones. This test cannot say which.
3. Neither side is ground truth. To settle individual disagreements, look at the 100-odd confident disagreements by hand (`model_vs_imputed_7377.csv`, filter `agree = False` and high confidence) with someone who knows the history.

## Caveats
- One model, one split (seed 42). The 2,161 voyages with neither value were not examined.
- Agreement depends on the model's predictors; the imputed column's method is unknown.
