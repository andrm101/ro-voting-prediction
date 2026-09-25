# roaep.ro Raw Data

All files in this folder are immutable originals. Do not edit.

Source: Autoritatea Electorală Permanentă (AEP) — https://www.roaep.ro/

## Confirmed election scope

### Primary target
| Election | Year | Level | Role in model |
|---|---|---|---|
| Parliamentary (Camera Deputaților + Senat) | Dec 2024 | Județ | Primary prediction target — party vote shares |

### Presidential — first rounds (climate check pair)
| Election | Year | Notes | Role |
|---|---|---|---|
| Presidential R1 | Nov 2024 | **Annulled** — Georgescu result | Baseline political climate; maps the protest/nationalist surge geographically |
| Presidential R1 | May 2025 | Redo election | Post-annulment landscape; Simion inherited much of Georgescu's geographic footprint |

Comparing these two first rounds county-by-county tests whether the nationalist vote was geographically stable or redistributed after the annulment. This is a key sanity-check and analytical layer in the project.

### Presidential — second rounds (ideological cleavage)
| Election | Year | Matchup | Role |
|---|---|---|---|
| Presidential R2 | May 2025 | Nicușor Dan vs George Simion | Binary pro-EU/reformist vs. nationalist/sovereigntist split — cleanest ideological signal |
| Presidential R2 | Nov 2019 | Iohannis vs Dăncilă | Historical reference; anti-PSD mobilisation, urban vs rural |

The R2 2025 binary is the sharpest ideological cleavage available in Romanian electoral history and will serve as a validation target for the socio-economic model.

### Supplementary (context, not primary targets)
| Election | Year | Notes |
|---|---|---|
| Local (consilii județene + primari) | Jun 2024 | Captures incumbency effects, local party machines |
| EP | Jun 2024 | Low-turnout protest vote signal, same day as local |
| Parliamentary | Dec 2020 | Previous wave — enables swing analysis 2020→2024 |

## Naming convention
`<type>_<year>_<round>_judet.csv`
Examples:
- `parlamentare_2024_judet.csv`
- `prezidentiale_2024_r1_judet.csv` (annulled)
- `prezidentiale_2025_r1_judet.csv`
- `prezidentiale_2025_r2_judet.csv`
- `prezidentiale_2019_r2_judet.csv`

## Key variables to extract per file
- `judet` — county name (standardise to SIRUTA codes for joining)
- `inscrisi_pe_liste` — registered voters
- `total_voturi_valabil_exprimate` — valid votes cast
- `<party_code>_voturi` — raw vote count per party/candidate
- Derived: `turnout = total_voturi_valabil_exprimate / inscrisi_pe_liste`
- Derived: `share_<party> = <party>_voturi / total_voturi_valabil_exprimate`

## Important notes
- Electoral data is at județ (NUTS 3). Do NOT average shares when aggregating — sum raw votes then compute share.
- București is split into 6 sectors in some roaep exports; re-aggregate to single București județ before joining with Eurostat.
- Sector 1–6 data should be summed to "Municipiul București" for NUTS 3 consistency.
- SIRUTA code for joining: use docs/siruta_nuts3_map.md.

## Citation format
Autoritatea Electorală Permanentă (<year>), Rezultate alegeri <type> <year>, accessed <date>, https://www.roaep.ro/
