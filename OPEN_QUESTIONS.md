# Open questions (for the consolidated session with Daniel)

Collected as they come up. Newest at the bottom. Mark each one answered when it is.

## About the data
1. **Imputed vs recorded pairs.** I paired columns by reading codebook section 3 (see `IMPUTATION_MAP` in `voyage_db.py`).
   Are these right? Least sure: `SLAXIMP` vs `TSLAVESD` and `SLAMIMP` vs `SLAARRIV` (the recorded counts are only approximate twins),
   and the three year pairs (`YEARDEP`/`DATEDEPC`, `YEARAF`/`DLSLATRC`, `YEARAM`/`DATARR34`).
2. **Imputed-only columns.** `MAJBYIMP`, `MAJBYIMP1`, `MJSELIMP`, `MJSELIMP1`, `TONMOD` have no recorded twin. In real mode they are all UNKNOWN.
   Should the regions instead be recomputed from the recorded place code using the geographic hierarchy?
3. **`NATIONAL` codes 3 and 6.** 21 voyages carry codes that the codebook only defines for the imputed column `NATINIMP`. A data-entry slip, or something intended?
4. **Undefined codes** (e.g. `RIG` 72/78/99, `FATE` 999, `XMIMPFLAG` eight codes on 167 voyages, a few port codes). Is there a note from the authors?
5. **How was `NATINIMP` filled** for the 7,377 voyages where `NATIONAL` is blank? From the port? From the flag of similar ships?
   If it was derived from a column I also use as a predictor, the imputed-mode scores are partly circular.
6. **`YEARDEP` is imputed but is also the best time axis.** I use it only to order voyages for the time-based split. OK?

## About the experiment
7. **Imputed-mode scores measure agreement with the authors' imputation, not with the truth.** Is that what we want to measure?
8. **Nationality accuracy is high (96% on random split) and not only because of the port.** Without any port column it is still 88%.
   Is that surprising to anyone who knows the data, or is it known that rig, year and tonnage separate the nations?
9. **The voyages that really lack a nationality look different** (they rarely have tonnage, build year or guns recorded), so the 96% probably overstates
   what we can do for them. Is there any way to get a small set of voyages whose nationality was found out later, to test on?
10. **Time-based split.** Nationality accuracy falls from 96% to about 38%. Training years (1526 to 1798) and test years (1816 to 1866) differ a lot.
    Is a time split the right stress test, or too harsh because Britain left the trade in 1807?

## Decisions I made that need a yes/no
- "Imputed" mode keeps recorded values untouched and only fills gaps. Agree?
- Targets for the first round: nationality (grouped to the 8 imputed-style groups) and tonnage. Which should come next?
