"""
Shared lookup tables used across ingest, clean, and feature scripts.

Single source of truth for:
  - SIRUTA codes (electoral/census join key)
  - NUTS 3 → NUTS 2 correspondence
  - SIRUTA → NUTS 3 Eurostat code mapping
  - Canonical party/candidate labels
  - Analytical composite groupings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SIRUTA codes — 42 NUTS 3 units (41 județe + București)
# ---------------------------------------------------------------------------
JUDET_SIRUTA: dict[str, int] = {
    "Alba": 10, "Arad": 20, "Argeș": 30, "Bacău": 40, "Bihor": 50,
    "Bistrița-Năsăud": 60, "Botoșani": 70, "Brașov": 80, "Brăila": 90,
    "Buzău": 100, "Caraș-Severin": 110, "Călărași": 120, "Cluj": 130,
    "Constanța": 140, "Covasna": 150, "Dâmbovița": 160, "Dolj": 170,
    "Galați": 180, "Giurgiu": 190, "Gorj": 200, "Harghita": 210,
    "Hunedoara": 220, "Ialomița": 230, "Iași": 240, "Ilfov": 250,
    "Maramureș": 260, "Mehedinți": 270, "Mureș": 280, "Neamț": 290,
    "Olt": 300, "Prahova": 310, "Satu Mare": 320, "Sălaj": 330,
    "Sibiu": 340, "Suceava": 350, "Teleorman": 360, "Timiș": 370,
    "Tulcea": 380, "Vaslui": 390, "Vâlcea": 400, "Vrancea": 410,
    "Municipiul București": 179, "București": 179,
}
SIRUTA_JUDET: dict[int, str] = {v: k for k, v in JUDET_SIRUTA.items()
                                  if k != "București"}

# ---------------------------------------------------------------------------
# NUTS 3 Eurostat codes → SIRUTA
# ---------------------------------------------------------------------------
NUTS3_SIRUTA: dict[str, int] = {
    "RO111": 50,  "RO112": 60,  "RO113": 130, "RO114": 260,
    "RO115": 320, "RO116": 330,
    "RO121": 10,  "RO122": 80,  "RO123": 150, "RO124": 210,
    "RO125": 280, "RO126": 340,
    "RO211": 40,  "RO212": 70,  "RO213": 240, "RO214": 290,
    "RO215": 350, "RO216": 390,
    "RO221": 90,  "RO222": 100, "RO223": 140, "RO224": 180,
    "RO225": 380, "RO226": 410,
    "RO311": 30,  "RO312": 120, "RO313": 160, "RO314": 190,
    "RO315": 230, "RO316": 310, "RO317": 360,
    "RO321": 179, "RO322": 250,
    "RO411": 170, "RO412": 200, "RO413": 270, "RO414": 300, "RO415": 400,
    "RO421": 20,  "RO422": 110, "RO423": 220, "RO424": 370,
}
SIRUTA_NUTS3: dict[int, str] = {v: k for k, v in NUTS3_SIRUTA.items()}

# ---------------------------------------------------------------------------
# NUTS 3 → NUTS 2 correspondence
# ---------------------------------------------------------------------------
NUTS3_NUTS2: dict[str, str] = {
    **{c: "RO11" for c in ["RO111","RO112","RO113","RO114","RO115","RO116"]},
    **{c: "RO12" for c in ["RO121","RO122","RO123","RO124","RO125","RO126"]},
    **{c: "RO21" for c in ["RO211","RO212","RO213","RO214","RO215","RO216"]},
    **{c: "RO22" for c in ["RO221","RO222","RO223","RO224","RO225","RO226"]},
    **{c: "RO31" for c in ["RO311","RO312","RO313","RO314","RO315","RO316","RO317"]},
    **{c: "RO32" for c in ["RO321","RO322"]},
    **{c: "RO41" for c in ["RO411","RO412","RO413","RO414","RO415"]},
    **{c: "RO42" for c in ["RO421","RO422","RO423","RO424"]},
}
NUTS2_NUTS3: dict[str, list[str]] = {}
for n3, n2 in NUTS3_NUTS2.items():
    NUTS2_NUTS3.setdefault(n2, []).append(n3)

# ---------------------------------------------------------------------------
# Canonical party / candidate labels
# Regex patterns matched in order — first match wins.
# ---------------------------------------------------------------------------
import re as _re

PARTY_PATTERNS: list[tuple[str, str]] = [
    # ── Parties ──────────────────────────────────────────────────────────────
    # PSD-PNL joint EP 2024 alliance — must come before individual party patterns
    (r"(?i)psd.*pnl|pnl.*psd|electoral[aă].*psd|electoral[aă].*pnl", "PSDPNL"),
    (r"(?i)(alian[tț]a|alianta).*(unire|aur)|a\.u\.r\.|^aur$",       "AUR"),
    (r"(?i)social.?democrat|p\.s\.d\.|^psd$",                         "PSD"),
    (r"(?i)na[tț]ional.?liberal|p\.n\.l\.|^pnl$",                    "PNL"),
    (r"(?i)salva[tț]i?.?rom[aâ]nia|u\.s\.r\.|^usr$",                 "USR"),
    (r"(?i)maghiar|u\.d\.m\.r\.|^udmr$|rmdsz",                       "UDMR"),
    (r"(?i)^sos\b|s\.o\.s\.",                                          "SOS"),
    (r"(?i)oamenilor.?tineri|p\.o\.t\.|^pot$",                        "POT"),
    (r"(?i)for[tț]a.?dreptei|f\.d\.",                                 "FD"),
    (r"(?i)re[iî]nnoim|reinnoim",                                      "REINNOIRE"),
    (r"(?i)pro.?rom[aâ]nia|^pro$",                                     "PRO"),
    # ── Presidential candidates ───────────────────────────────────────────
    (r"(?i)georgescu",                                                "GEORGESCU"),
    (r"(?i)simion",                                                   "SIMION"),
    (r"(?i)nicu[sș]or.?dan|nicusor.?dan",                            "DAN"),
    (r"(?i)antonescu",                                                "ANTONESCU"),
    (r"(?i)ponta",                                                    "PONTA"),
    (r"(?i)ciolacu",                                                  "CIOLACU"),
    (r"(?i)lasconi",                                                  "LASCONI"),
    (r"(?i)ciuc[aă]",                                                 "CIUCA"),
    (r"(?i)iohannis|johannis",                                        "IOHANNIS"),
    (r"(?i)d[aă]ncil[aă]",                                           "DANCILA"),
    (r"(?i)predoiu",                                                  "PREDOIU"),
    (r"(?i)funeriu",                                                  "FUNERIU"),
    (r"(?i)kelemen",                                                  "KELEMEN"),
]

_COMPILED_PATTERNS: list[tuple[_re.Pattern, str]] = [
    (_re.compile(p), label) for p, label in PARTY_PATTERNS
]


def normalise_party(raw: str) -> str:
    """Map a raw party/candidate string to its canonical label."""
    raw = str(raw).strip()
    for pattern, label in _COMPILED_PATTERNS:
        if pattern.search(raw):
            return label
    return "OTHER"


# ---------------------------------------------------------------------------
# Analytical composite groupings (parliamentary elections only)
# ---------------------------------------------------------------------------
ESTABLISHMENT_PARTIES = {"PSD", "PNL", "PSDPNL"}
ANTI_ESTABLISHMENT_PARTIES = {"AUR", "SOS", "POT", "GEORGESCU", "SIMION"}
CIVIC_PARTIES = {"USR", "FD", "REINNOIRE"}
ETHNIC_PARTIES = {"UDMR"}

# Parliamentary parties to model individually (primary targets)
PARLIAMENTARY_MODEL_PARTIES = ["AUR", "PSD", "PNL", "USR", "UDMR", "SOS", "POT"]

# Presidential candidates by election_id
PRESIDENTIAL_CANDIDATES: dict[str, list[str]] = {
    "prezidentiale_2024_r1": ["GEORGESCU", "CIOLACU", "LASCONI", "SIMION", "CIUCA"],
    "prezidentiale_2025_r1": ["SIMION", "DAN", "ANTONESCU", "PONTA"],
    "prezidentiale_2025_r2": ["DAN", "SIMION"],
    "prezidentiale_2019_r2": ["IOHANNIS", "DANCILA"],
}
