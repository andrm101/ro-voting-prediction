# RO-Voting-Prediction

ML-driven model predicting Romanian voting patterns at sub-national level using NUTS 2/3 socio-economic and demographic indicators from Eurostat and official election results from roaep.ro.

## Elections covered
- Presidential: 2019, 2024
- Parliamentary: 2020, 2024
- Local (county councils + mayors): 2020, 2024
- European Parliament: 2019, 2024

📖 **[Full abstract, 7 empirical findings, and hypothesis-testing methodology → project Wiki](https://github.com/andrm101/ro-voting-prediction/wiki)**

## Results at a glance

<p align="center">
  <img src="figures/06c_vote_profiles_final.png" width="48%" alt="Five stable voter-profile clusters" />
  <img src="figures/09a_swing_heatmap.png" width="48%" alt="AUR 2020-2024 swing heatmap" />
</p>

<p align="center">
  <img src="figures/10c_ger_ranked.png" width="48%" alt="Ghost electorate rate, ranked by county" />
  <img src="figures/07a_partial_corr_clusters.png" width="48%" alt="Georgescu-Simion bloc persistence, partial correlation" />
</p>

## Architecture

```mermaid
flowchart TD
    Eurostat["Eurostat NUTS2/3<br/>socio-economic data"] --> Ingest["01_ingest/"]
    Roaep["roaep.ro official<br/>election results"] --> Ingest
    Ingest --> Clean["02_clean/<br/>clean_*.parquet"]
    Clean --> Features["03_features/<br/>features_*.parquet"]
    Features --> Model["04_model/<br/>predictions_*.parquet"]
    Model --> Evaluate["05_evaluate/"]
    Evaluate --> Figures["figures/"]
    Evaluate --> Reports["reports/"]
```

## Pipeline stages
| Stage | Script(s) | Output |
|-------|-----------|--------|
| 01 Ingest | scripts/01_ingest/ | data/raw/ (immutable) |
| 02 Clean | scripts/02_clean/ | data/processed/clean_*.parquet |
| 03 Features | scripts/03_features/ | data/processed/features_*.parquet |
| 04 Model | scripts/04_model/ | data/processed/predictions_*.parquet |
| 05 Evaluate | scripts/05_evaluate/ | figures/, reports/ |

## Reproducing results
```bash
# 1. Activate environment
conda env create -f environment.yml
conda activate ro-voting

# 2. Run full pipeline (from project root)
python scripts/02_clean/clean_eurostat.py
python scripts/02_clean/clean_roaep.py
python scripts/03_features/build_features.py
python scripts/04_model/train_model.py
python scripts/05_evaluate/evaluate.py
```

All raw data must be downloaded manually — see data/raw/*/README.md for sources and download instructions.

## References
See references/references.bib for full bibliography.
