# assets/motion — code-drawn explainer video

`explainer.html` draws every frame as a pure function of time, `render(t)`, from
`data.json`. `render.mjs` captures the page frame by frame with Playwright and
encodes it with ffmpeg. Opened in a browser, the page simply plays.

```bash
python assets/motion/export_data.py                       # numbers from reports/*.csv + NUTS-3 outlines
npm i playwright                                          # once (any folder on NODE_PATH)
node assets/motion/render.mjs --name ro-voting-explainer  # needs ffmpeg on PATH (or $FFMPEG)
```

Outputs go to `docs/media/`: the MP4 (1280×720, H.264), a 640 px GIF preview of
the two map scenes and a poster frame.

| Time (s) | Scene | Source |
|---|---|---|
| 0–6 | Title | — |
| 6–18 | AUR swing 2020 → 2024 by county | `reports/09a_swing_2020_2024.csv` |
| 18–28 | Five voter profiles | `reports/10a_ghost_electorate.csv` (`cluster_short`) |
| 28–40 | Georgescu → Simion transfer | `reports/09b_presidential_arc.csv`, `13c_placebo_transfer.csv`, `07a_transfer_analysis.md` |
| 40–50 | Ghost electorate, top 12 counties | `reports/10a_ghost_electorate.csv` |
| 50–60 | Bayesian AUR swing, 90 % intervals | `reports/14a_bayes_predictions.csv` |
| 60–66 | Outro | — |

County outlines: `data/raw/eurostat/nuts3_2021_10M.geojson`. Prahova and Vâlcea
have no 2020 swing in `09a` and are hatched. `motion-kit.js` (helpers, brand
chrome, playback) and `motion.css` are shared with the other explainers in this
portfolio.
