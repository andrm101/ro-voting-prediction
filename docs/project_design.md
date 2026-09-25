# Project Design Document

## Research question
Which socio-economic, demographic, ethnic, and sentiment-proxy factors best predict county-level voting patterns in Romania, and do these factors explain the 2024–2025 anti-establishment electoral realignment?

## Unit of analysis
**NUTS 3 / județ** (N = 42: 41 județe + București)
Rationale: roaep.ro publishes at județ level; Eurostat NUTS 3 provides sufficient indicator coverage at 2022 baseline; NUTS 2 loses the intra-regional heterogeneity central to the research question.

---

## Political framing (context for model design)

### The establishment vs anti-establishment axis
Romanian politics 2024–2025 is best understood as a **two-axis system**:

**Axis 1 — Establishment / Governance bloc**
- PSD: dominant rural machine, clientelism, social spending, historically associated with state capture
- PNL: historical centre-right liberal party; long perceived as part of the PSD-PNL "uniparty" governing coalition; recently pivoted to align with USR on fiscal austerity measures in response to unsustainable public debt — this realignment is electorally costly in its traditional base but signals a governance-first positioning

**Axis 2 — Anti-establishment / Protest bloc**
- AUR: Romanian nationalist, sovereigntist, anti-NATO fringe in rhetoric, socially conservative
- SOS România: further right, Georgescu-adjacent
- POT: populist, Simion-connected
- Călin Georgescu (R1 2024 independent): zero-campaign, TikTok/Telegram phenomenon; won on pure anti-system sentiment and Orthodox-nationalist identity

**USR** sits in a third position: reform/civic, urban, pro-EU, anti-corruption — distinct from both axes above. The PNL-USR austerity alignment makes USR politically exposed: it is no longer in pure opposition but shares fiscal governance costs.

**UDMR**: ethnic party; independent of both axes. Backed by Orbán's Fidesz (Hungary). Will not align with AUR under any circumstances — AUR's Romanian ethno-nationalism is existentially incompatible with Hungarian minority political interests, regardless of any superficial policy overlap on social conservatism. Model UDMR separately as an ethnic bloc variable.

### The diaspora reversal
Romanian diaspora historically voted USR / pro-EU reform (2019 queues abroad were symbolic). In R1 Nov 2024, the diaspora voted ~44% Georgescu vs ~22% nationally — one of the most dramatic diaspora reversals in EU electoral history. Hypothesised drivers:
- Anti-establishment sentiment among economic migrants who observe Western European dysfunction
- TikTok/Telegram influence in diaspora communities not served by Romanian state media
- Disillusionment with PNL/USR governance performance despite (or because of) living abroad

Proxy for the model: **emigration rate per județ** (% of 2011 or 2021 population now abroad, from INS) — counties with high emigration may have structurally different voting because of diaspora influence on families at home.

### European macro context
The 2024 Romanian anti-establishment surge is not isolated: France (RN), Italy (FdI), Germany (AfD), Slovakia (Fico), Hungary (Fidesz already governing). This is a macro shock that lifts all nationalist/protest parties uniformly. It cannot be modelled at județ level (it affects all counties equally) but must be acknowledged as a **common shock** in the 2020→2024 panel comparison. Methodologically: include election-year fixed effects when running panel models; discuss macro wave in the limitations and context section.

---

## Election scope and analytical roles

### Primary target: Parliamentary Dec 2024
- Party vote shares (Camera Deputaților) at județ level
- Parties to model individually: **AUR, PSD, USR, UDMR, SOS, PNL**
- Composite targets: **Establishment bloc** (PSD + PNL combined share) vs **Anti-establishment bloc** (AUR + SOS + POT combined share)
- POT and smaller parties grouped into residual

### Layer 2: Presidential R1 pair — anti-establishment geography
| Election | Notes |
|---|---|
| Presidential R1 Nov 2024 (annulled) | Georgescu share — maps raw anti-system protest sentiment geographically |
| Presidential R1 May 2025 (redo) | Simion share — tests geographic stability of the nationalist vote post-annulment |

Cross-județ correlation between Georgescu 2024 and Simion 2025 is the headline diagnostic:
- High correlation → protest vote was geographically locked to structural factors
- Lower correlation → annulment reshuffled the vote (tactical, not structural)

### Layer 3: Presidential R2 2025 — ideological cleavage (binary)
- Dan vs Simion: pro-EU/urban/reformist vs nationalist/rural/sovereigntist
- Cleanest binary ideological signal in Romanian electoral history
- Compare with R2 2019 (Iohannis vs Dăncilă) to assess urban polarisation trend over 6 years

### Supplementary
- EP Jun 2024 + Local Jun 2024: use EP far-right share as a pre-parliamentary anti-establishment proxy
- Parliamentary Dec 2020: previous wave for swing analysis (2020→2024 change scores)

---

## Feature space

### Eurostat — 2022 baseline (NUTS 3 where available, NUTS 2 otherwise)
All Eurostat features pinned to **2022** as the most recent complete year before the 2024 elections. Justify in methods: socio-economic structure is relatively stable over 2–3 year horizons.

| Variable | Dataset | Mechanism |
|---|---|---|
| GDP per capita | nama_10r_3gdp / nama_10r_2gdp | Economic prosperity / grievance |
| Unemployment rate | lfst_r_lfu3rt | Labour precarity |
| At-risk-of-poverty rate | ilc_peps13 | Material deprivation |
| Tertiary education % (25–64) | edat_lfse_04 | Human capital / cosmopolitan values |
| Early school leaving % | edat_lfse_14 | Educational precarity |
| Internet access % | isoc_r_iuse_i | Digital inclusion; also TikTok access proxy |
| Employment in agriculture % | nama_10r_3empers | Rural/traditional economy |
| Population density | demo_r_d2jan | Urban–rural structural axis |
| % population >65 | demo_r_pjanaggr3 | Ageing — conservative/status-quo tendency |
| % population <25 | demo_r_pjanaggr3 | Youth — higher protest vote tendency |
| Net migration rate | demo_r_gind3 | Depopulation → anti-establishment; also diaspora connection |

### INS Census 2021 (județ level) — ethnic and religious foundational variables
These are treated as **structural/foundational** variables, not socio-economic controls. They explain bloc votes that are identity-driven rather than grievance-driven.

| Variable | Mechanism | Expected target |
|---|---|---|
| % Hungarian ethnicity | Ethnic bloc vote, Fidesz-UDMR connection | UDMR (near-linear) |
| % Roma ethnicity | Social marginalisation, PSD clientelism | PSD (hypothesised) |
| % Pentecostal | Conservative social values, anti-elite religiosity | AUR / Georgescu (hypothesised) |
| % Reformed/Calvinist | Magyar ethnic proxy | UDMR |
| % Greek Catholic | NW Transylvania historical; civic tradition | USR (hypothesised) |
| % Orthodox | Nearly universal; intensity varies by county | Low discriminating power alone; interact with rurality |
| % urban population | Reinforces density variable; urban = USR/civic | USR, Dan R2 |
| Emigration rate (INS) | Diaspora connection → anti-establishment transmission | AUR, Georgescu R1 |

### Sentiment proxies — structural and behavioural indicators
Direct survey data at județ level does not exist at sufficient granularity. The following are observable proxies for anti-establishment sentiment:

**1. EP 2024 far-right share (roaep.ro)**
AUR + SOS EP share per județ, June 2024 — six months before parliamentary elections. This is the cleanest pre-election anti-establishment signal. Use as a lagged dependent variable or feature depending on specification.

**2. 2020→2024 swing in anti-establishment vote**
Change in (AUR + SOS) share between parliamentary 2020 and 2024. Counties with the largest swing are where the macro-European anti-establishment wave landed hardest locally. This is a county-level measure of sentiment acceleration.

**3. Emigration rate (INS)**
Already listed above — doubles as both a demographic feature and a diaspora-sentiment transmission channel.

**4. Google Trends (manual download)**
Google Trends allows CSV export of search interest by region (NUTS 2 in Romania). Candidates: search interest in "Georgescu", "AUR", "corupție", "NATO" in Oct–Nov 2024 window. Limitation: NUTS 2 resolution only, not județ. Include as supplementary if coverage is sufficient; flag resolution mismatch.

**5. Barometrul Opiniei Publice (INSCOP / IRES)**
Romanian polling institutes (INSCOP Research, IRES) publish periodic survey waves with regional breakdowns. Some questions cover institutional trust, satisfaction with government, and economic pessimism. These surveys are downloadable as PDF/Excel from institute websites. If a 2023–2024 wave with regional data exists, institutional trust score per macroregion is a legitimate sentiment proxy. Limitation: macroregion (NUTS 1/2) resolution, not județ.

**6. Eurobarometer (European Commission)**
Special Eurobarometer surveys occasionally include Romania-specific modules. Standard EB asks about trust in national government, EU, and economic satisfaction. National level only — usable as a macro-context citation, not a county-level feature.

---

## Composite indices (feature engineering)

### Economic grievance index
Weighted composite of: unemployment rate + poverty rate + early school leaving % − GDP per capita (standardised, equal weights as baseline; test PCA-derived weights as sensitivity check).

### Human capital / modernity index
Tertiary education % + internet access % − early school leaving % (standardised).

### Demographic pressure index
Net out-migration rate + % population >65 − % population <25 (captures "shrinking, ageing county" syndrome associated with anti-establishment vote).

### Ethnic bloc intensity
% Hungarian + % Roma + % Pentecostal (each standardised). Not combined — kept separate. The Hungarian variable enters as a standalone structural control in all models to partial out UDMR.

---

## Modelling strategy

### Stage 1: Exploratory analysis
- Choropleth maps: AUR share, Georgescu R1, Simion R1, Dan R2 by județ
- Bivariate scatter: each feature vs AUR share
- Moran's I: test for spatial autocorrelation in parliamentary residuals
- Correlation matrix and VIF on full feature set
- PCA on Eurostat block to assess collinearity structure

### Stage 2: Baseline — Elastic Net (interpretable)
- One model per party share target
- Also run on establishment vs anti-establishment composite targets
- Report: coefficients, 95% CIs, VIF, adjusted R²
- Hungarian % enters all models to partial out ethnic bloc

### Stage 3: Tree ensemble — XGBoost (primary predictive)
- LOOCV (N=42 makes standard k-fold unreliable)
- SHAP values for feature attribution
- Spatial cross-validation: hold out one full NUTS 2 region at a time

### Stage 4: Spatial lag model (if Moran's I significant)
- PySAL / spreg; spatial weights matrix based on county contiguity
- Interpret spatial lag coefficient: magnitude of geographic spillover in voting

### Stage 5: Ideological cleavage binary
- Logistic: P(Dan wins județ) ~ f(features) for R2 2025
- Compare coefficient signs with parliamentary AUR model for consistency

### Stage 6: Temporal comparison
- Panel: 2020 + 2024 parliamentary, județ fixed effects, year dummy
- Identify which features drove the *change* in party shares, not just the level

---

## Key analytical hypotheses

1. **Grievance**: Unemployment + poverty → AUR/anti-establishment share (positive)
2. **Education**: Tertiary education % → USR share (positive), AUR share (negative)
3. **Depopulation**: Net out-migration → anti-establishment (positive); emigration rate → Georgescu/Simion share (positive)
4. **Ethnic bloc**: Hungarian % is the single strongest predictor of UDMR share (near-linear); Pentecostal % predicts AUR/Georgescu independently of economic deprivation
5. **Establishment persistence**: PSD share correlates with age structure and agricultural employment even after controlling for poverty (clientelism, not just deprivation)
6. **PNL erosion**: PNL share negatively associated with anti-establishment swing 2020→2024 in counties where it previously relied on PSD-adjacent rural networks
7. **Georgescu→Simion transfer**: județ-level r > 0.80 between Georgescu R1 2024 and Simion R1 2025 (geographic stability of nationalist vote)
8. **Diaspora reversal**: emigration rate positively associated with Georgescu/Simion share, contrary to the 2019 pro-USR diaspora pattern

---

## Structural constraints baked into the model

- UDMR modelled separately; never pooled with AUR in composite; UDMR swing probability ≈ 0 — use as a near-deterministic validation case
- Hungarian % enters every model as a forced non-penalised structural control (alpha=0 for that coefficient in Elastic Net)
- București-Ilfov (RO32) run with and without — report sensitivity
- Causal language forbidden throughout — all findings framed as *associated with* / *predicts*
- Annulled Nov 2024 R1 data cited explicitly as "annulled by Constitutional Court, Dec 2024" in all references

---

## ML robustness — synthetic data note
*(To be fully designed at modelling stage)*

N=42 is the binding constraint. Regularisation (Elastic Net, XGBoost with low `n_estimators` and high `min_child_weight`) partially addresses overfitting, but synthetic data augmentation should be explored:

- **Bootstrap resampling with noise injection**: generate synthetic counties by resampling real counties and adding Gaussian noise scaled to each feature's empirical variance. Preserves distributional shape without assuming a parametric model.
- **SMOTE-R (regression variant)**: for continuous targets, interpolate between nearest-neighbour counties in feature space. Risk: may smooth over the genuine geographic discontinuities that are substantively interesting (e.g., Harghita vs its neighbours).
- **Gaussian copula synthesis**: preserves inter-feature correlations while generating novel observations. More statistically principled than naive bootstrap.

Constraint: synthetic samples must be excluded from the test set in all evaluation. Cross-validation must be performed on real counties only. The purpose of synthetic data is regularisation aid, not performance inflation.

Decision: revisit this section when the real feature matrix is assembled and actual rank-deficiency / collinearity is measured.

---

## Interactive app (final deliverable)
Designed in Canva/Claude Design; implemented in React/JSX.

Views planned:
- Choropleth: vote share / predicted share per județ, switchable by election and party
- Feature explorer: any Eurostat/census variable vs party share scatter
- SHAP dashboard: county-level feature attribution from XGBoost
- Georgescu → Simion transfer: R1 2024 vs R1 2025 side-by-side with swing layer
- Ideological cleavage: R2 2025 binary map with model decision boundary

Python pipeline exports `features.json`, `predictions.json`, `shap_values.json` to `app/src/data/` as the interface between ML and frontend. See `app/README.md`.
