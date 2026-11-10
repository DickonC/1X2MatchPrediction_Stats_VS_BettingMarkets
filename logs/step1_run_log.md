# Step 1 run log (auto-generated)

```
Loaded Matches (238858, 48), EloRatings (273972, 4)
HomeTeam: stripped whitespace in 11 cells
HomeTeam: unified 235 team-name cells
AwayTeam: stripped whitespace in 32 cells
AwayTeam: unified 230 team-name cells
EloRatings: dropping 60 duplicate snapshot rows (['MGladbach', 'Nottm Forest'], 2025-06-15 to 2026-09-01)
EloRatings: unified 293 club-name cells

EPL matches: 9810 (2000/2001 to 2026/2027)
EPL matches per season with != 380 games: {'2003/2004': 335, '2004/2005': 335, '2026/2027': 20}
Form3Home: corrected 1 cells
Form5Home: corrected 1 cells
Form3Away: corrected 0 cells
Form5Away: corrected 1 cells
HomeElo: filled 1 missing cells
AwayElo: filled 1 missing cells

Dropped 51 rows: burn-in: a team has < 5 earlier matches in dataset (form/rolling stats unreliable)
Dropped 1779 rows: season outside the train/val/test range (2005/2006 to 2025/2026)
Dropped 0 rows: missing target

Final dataset: 7980 matches x 67 columns -> data_processed/epl_step1.csv

Rows per split / class balance:
FTResult     A     D     H   All
Split                           
test       246   197   317   760
train     2034  1662  3144  6840
val        123    82   175   380
All       2403  1941  3636  7980

FTResult      A      D      H
Split                        
test      0.324  0.259  0.417
train     0.297  0.243  0.460
val       0.324  0.216  0.461

Columns with missing values (% of rows):
HandiSize        0.1
HandiHome        0.1
HandiAway        0.1
Home_H2H_PPG     2.4
Home_H2H_GDPG    2.4

Missing values by split for odds columns that are absent in early seasons:
       OddHome  MaxHome  Over25  HandiSize
Split                                     
test       0.0      0.0     0.0      0.001
train      0.0      0.0     0.0      0.001
val        0.0      0.0     0.0      0.000

Total logged cell changes: 806
```
