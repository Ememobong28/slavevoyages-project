# Tonnage: percent error (step 1)

Tonnage ranges from 10 to 1,667 tons (median 129). An error of 17 tons means little for a 600-ton ship and a lot for a 40-ton one, so the table reports error as a **percent of the true tonnage**.

- **Median percent error:** the typical miss, as a percent of the true tonnage.
- **Within 25%:** the share of test voyages predicted within 25% of the true tonnage. Read it like an accuracy.
- Average over 5 random splits (70% of known values for training). The time-based split has one fixed split, so it has no spread.
- Test set: 2,651 voyages. TabPFN was run in Colab (notebook section 6) on the same voyages; its predictions are in `tabpfn_tonnage/`.

| Split | Model | Median % error | Within 10% | Within 25% | Within 50% | Average miss (tons) |
| --- | --- | --- | --- | --- | --- | --- |
| Random | Median guess | 39.7 | 12.9% | 27.8% | 60.9% | 74.3 |
| Random | Decision tree | 20.0 | 30.7% | 58.0% | 82.5% | 43.2 |
| Random | Gradient boosting | 17.0 | 31.9% | 64.5% | 87.5% | 36.9 |
| Random | Random forest | 16.6 | 35.6% | 64.3% | 87.3% | 36.8 |
| Random | TabPFN | 14.4 | 39.9% | 69.2% | 89.9% | 32.1 |
| Time-based | Median guess | 36.4 | 14.6% | 33.5% | 69.9% | 67.6 |
| Time-based | Decision tree | 52.4 | 9.3% | 24.2% | 47.8% | 93.4 |
| Time-based | Gradient boosting | 49.8 | 10.5% | 26.1% | 50.3% | 97.2 |
| Time-based | Random forest | 55.3 | 9.5% | 24.1% | 45.8% | 101.2 |
| Time-based | TabPFN | 46.4 | 10.0% | 27.8% | 52.9% | 85.7 |

(Predictors: recorded values only. The same table with imputed values filled in is in `tonnage_relative_error_table.csv`.)

## What it says
0. **TabPFN is best on the random split** (typical error 14.4%, 69.2% within 25%). On the time-based split it is the best of the four models but still worse than the median guess (46.4% vs 36.4%).
1. **On a random split, the models roughly halve the typical error:** about 17% instead of 40% for always guessing the median. Within 25%, they are right about 64% of the time against 28%.
2. **That is much weaker than nationality's 96%.** "Right within 25%" for tonnage is the number to put next to nationality's accuracy.
3. **On the time-based split, every model is worse than guessing the median** (typical error 50% or more, against 36%). Models trained on earlier voyages do not carry over to later ones.
4. **Filling in imputed values changes little on the random split and hurts on the time-based split.** With imputed predictors, the decision tree's average miss rises from 93 to 135 tons. I have not found out why.

## Limits
Spread across random splits is small (median % error varies by under 2 points). Mean percent error (`MAPE_pct` in the files) is pulled up by very small ships; use the median.
