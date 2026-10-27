# Getting the data

This repository does **not** contain the SlaveVoyages data. You need two things:

1. `tastdb-exp-2019.csv`: the 2019 SlaveVoyages trans-Atlantic voyages export (36,108 voyages, 276 columns), from <https://www.slavevoyages.org/>. Check the site's terms of use before sharing copies.
2. (optional) the codebook PDF `SPSS_Codebook_2019.pdf` from the same place. It explains every column and code. The labels used here were taken from it and are in `codebook_maps.json`.

Put the CSV in the repo root (or `data/`), then:

```
python voyage_db.py sqlite tastdb-exp-2019.csv voyages.sqlite
```

The original CSV is only ever read, never changed.
