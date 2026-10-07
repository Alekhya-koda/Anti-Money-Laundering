# AML Account Risk Ranking (IBM AML dataset, HI-Small)

## Project goal
Banks in Singapore must monitor transactions under MAS Notice 626. Fixed rules (e.g. "flag anything above X") create many false alerts and miss new laundering patterns. Investigators have limited time, so they need the **riskiest accounts ranked first**. The system must be **explainable** (MAS FEAT: Fairness, Ethics, Accountability, Transparency).

**Deliverable:** a ranked list of accounts by laundering risk, with human-readable reasons for each.

## Folder contents
```
aml_project/
  README.md                          <- this file
  silver/transactions.parquet        <- cleaned transactions (one row per txn)
  gold/account_features.parquet      <- v1 account features (EDA only, do NOT model on this)
  gold_v2/gold_v2.parquet            <- shared modelling table (USE THIS)
  baselines_v2.csv                   <- v2 baselines (test snapshot): the benchmark to beat
```
The folder is **view-only**. Add it to your Drive via right-click > Organize > Add shortcut to Drive, then save your own work in your **own** folder.

Load the data in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
g = pd.read_parquet('/content/drive/MyDrive/aml_project/gold_v2/gold_v2.parquet')
```

## Dataset
- Source: IBM synthetic AML transactions (Altman et al.), **HI-Small** variant.
- 5,078,336 transactions, about 515k accounts, 15 currencies.
- Illicit rate: **0.10% of transactions**, **0.655% of accounts** (full period).
- 370 laundering attempts across 8 pattern types: cycle, gather-scatter, bipartite, fan-out, scatter-gather, stack, random, fan-in.
- The data is synthetic and not Singapore-specific. State this in the limitations.

## Data layers
| Layer | What it is |
|---|---|
| Bronze | Raw CSV as parquet, all text (not shared, same as the Kaggle file) |
| Silver | Typed, deduplicated transactions with flags: cross-bank, cross-currency, self-transfer, hour of day |
| Gold v1 | One row per account over the full period. Used for EDA. **Leaks** because features overlap the label period. |
| **Gold v2** | One row per account per snapshot. Features from the past, label from the following 2 days. **Model on this.** |

## Gold v2 design
| Snapshot | Features from | Label from |
|---|---|---|
| train | Sep 1-4 | Sep 5-6 |
| valid | Sep 3-6 | Sep 7-8 |
| test | Sep 5-8 | Sep 9-10 |

- **Label** = the account *sent* at least one laundering transaction in the label window.
- **Window:** Sep 1-10 only.
- **Alert universe:** self-transfers and cross-currency transactions are excluded (about 0% illicit rate).
- **Amounts:** `amt_pct` is the percentile of the amount within its currency. Raw amounts are not comparable across currencies.
- No feature uses `is_laundering`.

## Rules for everyone
1. Train on `snapshot == 'train'`, tune on `valid`, evaluate on `test` **once**, at the end.
2. Do not add columns to the shared files. Build extra features in your own notebook and join on `snapshot` + `acct`.
3. Do not compare raw amounts across currencies. Use `amt_pct`.
4. Never report accuracy. Report **Precision@K, Recall@K (K = 500, 2000, 5000) and PR-AUC**.
5. Compare models within the same snapshot. The positive rate drifts upward across snapshots.
6. Do not evaluate on Sep 11-18 (see below).

## Findings that affect everyone
- **Fixed rules are weak.** "Amount > 10,000" gave 0.176% precision with 1.37M alerts, about 1.7x random.
- **Simple account ranking is the benchmark.** Ranking by one feature (e.g. `out_cnt`) gives about 13x lift at the top 500, but finds only about 5% of illicit accounts in the top 5,000. See `baselines_v2.csv`.
- **Time trap:** after Sep 10 legitimate volume collapses (about 480k/day to a few hundred) while laundering chains continue. After Sep 10, 59% of transactions are illicit. Including this tail makes the task look artificially easy.
- **ACH dominates:** ACH carries 87% of illicit transactions (0.75% rate, about 7x base). Wire and Reinvestment have zero. This is a quirk of the synthetic data, so models will lean on it. Report it as a limitation and watch it in SHAP.
- **Self-transfers and cross-currency** transactions have about 0% illicit rate.
- **Open check:** `out_cross_cur_share` showed AUC 0.71 in v1 even though cross-currency transactions are almost never illicit. This was not explained yet.
- **Coverage ceiling:** illicit senders with no history in the feature window cannot be caught by any model. See the coverage printout in the notebook.


## Code and collaboration
- Code in GitHub, one branch per person, **no data in the repo** (`data/` in `.gitignore`).
- Docker (or `requirements.txt`) gives the same environment for everyone. It does not share data, so data lives in the Drive folder.
- Only one person (the data engineer) writes to the shared folder.# Anti-Money-Laundering
