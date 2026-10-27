# Query examples

All of these run as written on `voyages.sqlite` (build it with `python voyage_db.py sqlite tastdb-exp-2019.csv voyages.sqlite`). Output is from the 2019 data.

Switching modes is just switching the table name: `voyages_real` or `voyages_imputed`. Missing is `NULL`. In the two views, `YEARDEP` follows the mode (the real one is the recorded year, which is often missing); the `voyages` table keeps the original complete imputed `YEARDEP`.

## How many voyages for each recorded nationality?
```sql
SELECT l.label AS nationality, COUNT(*) AS voyages FROM voyages_real v JOIN category_labels l ON l.column_name = 'NATIONAL' AND l.code = v.NATIONAL GROUP BY 1 ORDER BY 2 DESC LIMIT 5;
```
Result:

```
  nationality  voyages
Great Britain    11239
     Portugal     5332
       France     4090
       U.S.A.     1799
        Spain     1637
```

## Known departure ports in each mode
```sql
SELECT (SELECT COUNT(PORTDEP) FROM voyages_real) AS real_mode, (SELECT COUNT(PORTDEP) FROM voyages_imputed) AS imputed_mode;
```
Result:

```
 real_mode  imputed_mode
     29158         31635
```

## Which voyages get a departure port only from imputation?
```sql
SELECT VOYAGEID, PTDEPIMP AS imputed_port FROM voyages WHERE PORTDEP IS NULL AND PTDEPIMP IS NOT NULL LIMIT 5;
```
Result:

```
 VOYAGEID  imputed_port
        5       50299.0
        6       50299.0
        7       50299.0
        9       50299.0
       11       50299.0
```

## Most common departure ports, with names (imputed mode)
```sql
SELECT p.label AS port, COUNT(*) AS voyages FROM voyages_imputed v JOIN place_labels p ON p.code = v.PORTDEP GROUP BY 1 ORDER BY 2 DESC LIMIT 5;
```
Result:

```
                   port  voyages
              Liverpool     4975
Bahia, port unspecified     4478
                 London     3125
         Rio de Janeiro     2545
                Bristol     2082
```

## Share of voyages missing tonnage, by 50-year period (`voyages` keeps the complete YEARDEP)
```sql
SELECT (CAST(YEARDEP AS INTEGER)/50)*50 AS period, ROUND(100.0*SUM(TONNAGE IS NULL)/COUNT(*),1) AS pct_missing, COUNT(*) AS voyages FROM voyages GROUP BY 1 ORDER BY 1 LIMIT 6;
```
Result:

```
 period  pct_missing  voyages
   1500        100.0       91
   1550         95.5      755
   1600         67.9     1247
   1650         76.4     2164
   1700         51.3     7898
   1750         32.9    12883
```

## How many columns got each audit decision?
```sql
SELECT substr(Decision,1,instr(Decision||' ',' ')-1) AS decision, COUNT(*) AS columns FROM column_audit GROUP BY 1 ORDER BY 2 DESC;
```
Result:

```
decision  columns
   DEFER       93
 RESHAPE       77
 EXCLUDE       73
    KEEP       33
```

## The pairs used to fill gaps
```sql
SELECT concept, recorded_column, imputed_column, confidence FROM imputation_map;
```
Result:

```
 concept recorded_column imputed_column confidence
 PORTDEP         PORTDEP       PTDEPIMP       high
MAJBUYPT        MAJBUYPT      MJBYPTIMP       high
MAJSELPT        MAJSELPT      MJSLPTIMP       high
 YEARDEP        DATEDEPC        YEARDEP     medium
  YEARAF        DLSLATRC         YEARAF     medium
  YEARAM        DATARR34         YEARAM     medium
 SLAXIMP        TSLAVESD        SLAXIMP        low
 SLAMIMP        SLAARRIV        SLAMIMP        low
```
