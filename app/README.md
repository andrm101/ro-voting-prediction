# RO-Voting-Prediction — Interactive App

React/JSX single-page application for exploring model outputs interactively.
Design to be produced in Canva/Claude Design; implementation in JSX.

## Planned views
- **Choropleth map**: vote share or predicted share per județ, switchable by election/party
- **Feature explorer**: scatter plot of any Eurostat/census feature vs. party share
- **SHAP dashboard**: county-level feature attribution from XGBoost model
- **Georgescu → Simion transfer**: side-by-side R1 2024 vs R1 2025 maps with swing layer
- **Ideological cleavage**: R2 2025 Dan/Simion binary map with decision boundary overlay

## Data interface
Python pipeline exports prediction results and feature data to `app/src/data/`:
- `features.json` — all county features (42 rows)
- `predictions.json` — model outputs per party per county
- `shap_values.json` — SHAP attribution per county per feature
- `geojson/ro_judete.geojson` — county boundaries for choropleth

## Stack
- React 18 + Vite
- D3.js or react-simple-maps for choropleth
- Recharts for scatter/bar charts
- Tailwind CSS for layout

## Setup (after design phase)
```bash
cd app
npm install
npm run dev
```
