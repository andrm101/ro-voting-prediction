"""
export_data.py — every number the explainer video shows, read from the
project's own outputs (reports/*.csv) and the Eurostat NUTS-3 outlines.

    python assets/motion/export_data.py      → assets/motion/data.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REP = os.path.join(ROOT, "reports")
W, H = 640, 470                         # map box in the video (px)


def key(name: str) -> str:
    """county key that ignores diacritics (ş/ș, ţ/ț) and case"""
    s = unicodedata.normalize("NFKD", name.replace("ş", "s").replace("ţ", "t"))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().replace(" ", "-")


def rows(name):
    with open(os.path.join(REP, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def r(x, n=4):
    return None if x in (None, "") else round(float(x), n)


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def county_paths():
    with open(os.path.join(ROOT, "data", "raw", "eurostat", "nuts3_2021_10M.geojson"), encoding="utf-8") as fh:
        geo = json.load(fh)
    feats = [f for f in geo["features"]
             if f["properties"]["CNTR_CODE"] == "RO" and f["properties"]["LEVL_CODE"] == 3]
    k = math.cos(math.radians(46))
    polys = []
    for f in feats:
        g = f["geometry"]
        ps = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        polys.append((f["properties"]["NUTS_NAME"], [[(x * k, -y) for x, y in ring] for p in ps for ring in p]))
    xs = [x for _, rs in polys for ring in rs for x, _ in ring]
    ys = [y for _, rs in polys for ring in rs for _, y in ring]
    x0, y0 = min(xs), min(ys)
    s = min(W / (max(xs) - x0), H / (max(ys) - y0))
    return {key(n): "".join("M" + "L".join(f"{s * (x - x0):.1f},{s * (y - y0):.1f}" for x, y in ring) + "Z"
                            for ring in rs) for n, rs in polys}


def main():
    paths = county_paths()
    swing = {key(x["judet"]): x for x in rows("09a_swing_2020_2024.csv")}
    arc = rows("09b_presidential_arc.csv")
    ghost = rows("10a_ghost_electorate.csv")
    bayes = rows("14a_bayes_predictions.csv")
    transfer = next(x for x in rows("13c_placebo_transfer.csv") if x["description"].startswith("Nationalist continuity"))
    with open(os.path.join(REP, "07a_transfer_analysis.md"), encoding="utf-8") as fh:
        md = fh.read()
    ci = [float(v) for v in re.search(r"95% bootstrap CI: \[([\d.]+), ([\d.]+)\]", md).groups()]
    rv = float(re.search(r"Robustness Value RV = ([\d.]+)", md).group(1))

    cluster = {key(x["judet"]): x["cluster_short"] for x in ghost}
    counties = []
    for k_, d in paths.items():
        s = swing.get(k_) or swing.get("MUNICIPIUL-" + k_)   # 09a spells the capital in full
        counties.append({"key": k_, "d": d, "sw_aur": r(s["sw_AUR"]) if s else None,   # no 2020 data: Prahova, Vâlcea
                         "cluster": cluster.get(k_) or cluster.get("MUNICIPIUL-" + k_)})
    g1, s1 = [float(x["georgescu_r1"]) for x in arc], [float(x["simion_r1"]) for x in arc]
    inside = sum(float(b["pp_q05"]) <= float(b["sw_AUR"]) <= float(b["pp_q95"]) for b in bayes)
    data = {
        "map": {"w": W, "h": H, "counties": counties},
        "clusters": sorted({c["cluster"] for c in counties if c["cluster"]}),
        "transfer": {"points": [{"c": x["judet"], "g": r(x["georgescu_r1"]), "s": r(x["simion_r1"])} for x in arc],
                     "r_raw": round(pearson(g1, s1), 3), "r_partial": r(transfer["r_partial"], 3),
                     "ci": ci, "rv": rv, "n": int(transfer["n"])},
        "ghost": sorted(({"c": x["judet"], "ger": r(x["ger"])} for x in ghost), key=lambda x: -x["ger"]),
        "bayes": sorted(({"c": x["judet"], "y": r(x["sw_AUR"]), "m": r(x["pp_mean"]), "lo": r(x["pp_q05"]),
                          "hi": r(x["pp_q95"])} for x in bayes), key=lambda x: x["y"]),
        "bayes_inside": inside,
    }
    out = os.path.join(HERE, "data.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
    sw = [c["sw_aur"] for c in counties if c["sw_aur"] is not None]
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out) // 1024} kB)")
    print(len(counties), "counties,", len(sw), "with swing", min(sw), max(sw), data["clusters"])
    print({k_: v for k_, v in data["transfer"].items() if k_ != "points"}, "inside 90%:", inside, "/", len(bayes))
    print(data["ghost"][:3], data["ghost"][-3:])


if __name__ == "__main__":
    main()
