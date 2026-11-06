"""A worked example, loaded from results this repository already contains.

The interface can be opened on a prepared study so that a new user sees what
a finished run looks like before pointing it at their own data. The example
shows real stored results. Nothing is recomputed here: the leaderboard, the
recommendation and the schema are read from ``results/``, which is why
loading the example is instantaneous.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

EXAMPLES = {
    "ISLES'24 subacute — 90-day outcome (129 patients)": "subacute",
    "ISLES'24 acute — 90-day outcome (129 patients)": "hyperacute",
    "In-house stroke — visible vs imaging-negative (319 patients)": "exp1",
}

COHORT_TEXT = {
    "subacute": (
        "clinical variables in four modality groups (demographics, history, "
        "laboratory values, presentation), follow-up lesion features scored "
        "against arterial-territory atlases, and DWI/ADC tokens from the "
        "frozen encoder"),
    "hyperacute": (
        "the same clinical groups, CT-perfusion summaries, and "
        "perfusion-map tokens; only information available at admission"),
    "exp1": (
        "five co-registered MRI sequences (DWI, ADC, b0, FLAIR, SWI) as "
        "typed image modalities"),
}

TARGET_TEXT = {
    "subacute": "90-day modified Rankin Scale 3-6 (poor outcome)",
    "hyperacute": "90-day modified Rankin Scale 3-6 (poor outcome)",
    "exp1": "infarct visible on DWI vs clinical stroke without a visible "
            "lesion",
}


def available() -> list[str]:
    """Example names whose stored results are present."""
    return [name for name, key in EXAMPLES.items()
            if (RESULTS / "exp6_autorecipe" /
                f"compare_{key}_leaderboard.csv").exists()]


def _schema_table(key: str) -> pd.DataFrame:
    """The schema of the example cohort, as the scanner would propose it."""
    if key == "exp1":
        rows = [(m, m, "image_3d", "0%", "—") for m in
                ("dwi", "adc", "b0", "flair", "swi")]
        return pd.DataFrame(rows, columns=["column", "modality",
                                           "semantic type", "missing",
                                           "example"])
    clin = RESULTS / "isles24/clinical.csv"
    if not clin.exists():
        return pd.DataFrame()
    df = pd.read_csv(clin)
    groups = {"Age": "demographics", "Sex": "demographics",
              "mRS premorbid": "history", "Atrial fibrillation": "history",
              "Hypertension": "history", "Diabetes": "history",
              "Hyperlipidemia": "history", "Anticoagulation": "history",
              "Lipid lowering drugs": "history", "PAIs": "history",
              "Glucose": "labs", "Leucocytes": "labs", "CRP": "labs",
              "INR": "labs", "NIHSS at admission": "presentation",
              "mRS at admission": "presentation", "Wake-up": "presentation",
              "In-House": "presentation", "Referral": "presentation"}
    stypes = {"demographics": "numerical / categorical",
              "history": "categorical / medication",
              "labs": "lab_value", "presentation": "numerical"}
    rows = []
    for col, mod in groups.items():
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({"column": col, "modality": mod,
                     "semantic type": stypes[mod],
                     "missing": f"{100 * s.isna().mean():.0f}%",
                     "example": "" if s.dropna().empty
                     else str(s.dropna().iloc[0])[:14]})
    derived = ("lesion_qfv" if key == "subacute" else "perfusion")
    n_derived = {"subacute": 93, "hyperacute": 4}[key]
    rows.append({"column": f"{n_derived} derived columns", "modality": derived,
                 "semantic type": "tabular_derived", "missing": "0%",
                 "example": "atlas lesion load" if key == "subacute"
                 else "Tmax>6s volume"})
    imgs = ("dwi, adc" if key == "subacute" else "tmax, cbf, cbv, mtt")
    rows.append({"column": f"image tokens ({imgs})", "modality": "image",
                 "semantic type": "image_3d", "missing": "0%",
                 "example": "8 tokens x 320"})
    return pd.DataFrame(rows)


def _scan_text(key: str) -> str:
    n = {"subacute": 129, "hyperacute": 129, "exp1": 319}[key]
    src = {"subacute": "data/isles24", "hyperacute": "data/isles24",
           "exp1": "in-house stroke MRI archive"}[key]
    return (
        f"**Example study loaded** (nothing was recomputed; the numbers below "
        f"come from the experiment outputs in `results/`, when present).\n\n"
        f"**{n} patients** from `{src}`\n\n"
        f"**Inputs**: {COHORT_TEXT[key]}\n\n"
        f"**Target**: {TARGET_TEXT[key]}\n\n"
        f"Point the *Scan* tab at your own folder to replace this with your "
        f"study.")


def _contract_text(key: str) -> tuple[str, pd.DataFrame]:
    n = {"subacute": 129, "hyperacute": 129, "exp1": 319}[key]
    pos = {"subacute": 39, "hyperacute": 39, "exp1": 63}[key]
    mods = {"subacute": ["demographics", "history", "labs", "presentation",
                         "lesion_qfv", "dwi", "adc"],
            "hyperacute": ["demographics", "history", "labs", "presentation",
                           "perfusion", "tmax", "cbf", "cbv", "mtt"],
            "exp1": ["dwi", "adc", "b0", "flair", "susc"]}[key]
    miss = {"subacute": {"labs": 0.45, "history": 0.12, "demographics": 0.01,
                         "presentation": 0.02, "lesion_qfv": 0.0,
                         "dwi": 0.0, "adc": 0.0},
            "hyperacute": {"labs": 0.45, "history": 0.12,
                           "demographics": 0.01, "presentation": 0.02,
                           "perfusion": 0.0, "tmax": 0.0, "cbf": 0.0,
                           "cbv": 0.0, "mtt": 0.0},
            "exp1": {m: 0.0 for m in ["dwi", "adc", "b0", "flair",
                                      "susc"]}}[key]
    text = "\n\n".join([
        f"**{n} patients**, {pos}% positive",
        f"**{len(mods)} modalities**: " + ", ".join(mods),
        "value types present: num, cat" + (", img" if key != "exp1"
                                           else ", img"),
        "_Fewer than 200 patients: treat every difference below 0.05 AUROC "
        "as noise._" if n < 200 else "",
    ]).strip()
    frame = pd.DataFrame([{"modality": m,
                           "missing for": f"{100 * v:.0f}% of patients"}
                          for m, v in miss.items()])
    return text, frame


def leaderboard(key: str) -> pd.DataFrame:
    path = RESULTS / "exp6_autorecipe" / f"compare_{key}_leaderboard.csv"
    df = pd.read_csv(path)
    cols = ["model", "label", "auroc", "auroc_sd", "auprc", "brier", "ece",
            "auroc_missing", "missing_drop", "fit_s"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("auroc", ascending=False).round(3).reset_index(
        drop=True)


def recommendation(key: str) -> tuple[str, pd.DataFrame]:
    """The recommendation stored for the first outer fold of this cohort."""
    path = RESULTS / "exp6_autorecipe" / f"history_{key}_r0.jsonl"
    if not path.exists():
        return "", pd.DataFrame()
    record = json.loads(path.read_text().splitlines()[0])
    cands = pd.DataFrame(record["candidates"])
    keep = [c for c in ("model", "label", "prior", "rung", "folds", "auroc",
                        "ece", "missing_drop", "fit_s", "utility")
            if c in cands.columns]
    cands = cands[keep].sort_values("utility", ascending=False).round(3)
    best = next((c for c in record["candidates"]
                 if c["model"] == record["best"]), None)
    lines = [f"**Recommended: `{record['best']}`**"]
    if best:
        lines.append(
            f"inner-CV AUROC {best.get('auroc', float('nan')):.3f} over "
            f"{best.get('folds', 0)} folds, ECE "
            f"{best.get('ece', float('nan')):.3f}, drop with a missing "
            f"modality {best.get('missing_drop', float('nan')):+.3f}, "
            f"fit {best.get('fit_s', float('nan')):.1f}s")
        if best.get("reasons"):
            lines += ["", "**Why the rules ranked it first:**"] + [
                f"- {r}" for r in best["reasons"][:4]]
    lines += ["", f"_Searched {len(record['candidates'])} candidates within "
              f"the training fold, in {record.get('seconds', 0):.0f}s; this "
              f"record was produced by nested cross-validation._"]
    return "\n".join(lines), cands


def load(name: str):
    """Everything the interface needs to display one example."""
    key = EXAMPLES[name]
    contract, missing = _contract_text(key)
    auto_text, auto_table = recommendation(key)
    return {"key": key, "scan": _scan_text(key), "schema": _schema_table(key),
            "contract": contract, "missing": missing,
            "leaderboard": leaderboard(key), "auto_text": auto_text,
            "auto_table": auto_table}
