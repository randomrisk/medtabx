"""MedTabX Studio: a browser interface over the typed contract.

Intended for a clinician or student who has a study folder and no wish to
write code. The workflow has five steps:

  1. Scan       point at a folder; the app finds subjects, imaging
                modalities and tables, and proposes a schema.
  2. Schema     review and edit the proposal: which column belongs to which
                modality, what it means, what is being predicted.
  3. Contract   build the typed contract and look at it: how many patients,
                which modalities, how much is missing.
  4. Run        compare models, or let AutoRecipe choose one, on identical
                folds with test-time missing-modality scenarios.
  5. Export     save the leaderboard, the recipe and a script that
                reproduces the run outside the interface.

Nothing starts computing until a button is pressed, and every long step
reports progress. Launch with::

    python -m medtabx.app            # then open the printed URL
    python -m medtabx.app --share    # temporary public link
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import gradio as gr

from medtabx.app import demo as demo_data
from medtabx.app.scan import ScanResult, scan_study

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

INTRO = """
# MedTabX Studio

Point at a study folder. The app proposes a typed schema, builds the
contract, and runs one comparison protocol: every model on
identical folds, scored both with all inputs and with one modality removed.

*Nothing runs until you press a button. Fitting happens on this machine.*
"""

STATE_HELP = """**Nothing scanned yet.** Enter a folder and press *Scan
folder*. A BIDS-style tree works out of the box; so does one folder per
subject with the sequences inside."""


# --------------------------------------------------------------------------
# state helpers
# --------------------------------------------------------------------------


class Session:
    """Everything the interface has learned so far, in one object."""

    def __init__(self) -> None:
        self.scan: ScanResult | None = None
        self.columns: pd.DataFrame | None = None
        self.data = None            # TypedData
        self.builder = None
        self.meta: dict = {}
        self.rows: pd.DataFrame | None = None
        self.leaderboard: pd.DataFrame | None = None
        self.recipe: dict | None = None


def _err(exc: Exception) -> str:
    return (f"**Something went wrong.**\n\n```\n{type(exc).__name__}: {exc}\n"
            f"```\n\n<details><summary>Details</summary>\n\n```\n"
            f"{traceback.format_exc(limit=4)}\n```\n</details>")


# --------------------------------------------------------------------------
# step 1: scan
# --------------------------------------------------------------------------


def do_scan(folder: str, session: Session, progress=gr.Progress()):
    if not folder or not folder.strip():
        return (session, STATE_HELP, None, None, gr.update(choices=[]),
                gr.update(choices=[]))
    try:
        progress(0.1, desc="walking the folder")
        result = scan_study(folder.strip())
        session.scan = result
        session.columns = result.column_table()
        progress(0.9, desc="proposing a schema")
        targets = result.candidate_targets or (
            [c.name for c in result.columns] if result.columns else [])
        modalities = sorted(result.image_modalities) + sorted(
            {c.modality for c in result.columns})
        return (session, result.summary(), result.modality_table(),
                session.columns,
                gr.update(choices=targets,
                          value=targets[0] if targets else None),
                gr.update(choices=modalities, value=modalities))
    except Exception as exc:  # noqa: BLE001 - surfaced in the interface
        return (session, _err(exc), None, None, gr.update(choices=[]),
                gr.update(choices=[]))


def load_example(name: str, session: Session):
    """Fill every tab from results stored in the repository."""
    try:
        ex = demo_data.load(name)
        session.leaderboard = ex["leaderboard"].set_index("model")
        session.rows = None
        session.recipe = None
        board = ex["leaderboard"]
        note = (f"**Precomputed example.** {len(board)} models on identical "
                f"folds; nothing was fitted just now. Best on full inputs: "
                f"`{board.iloc[0]['model']}` ({board.iloc[0]['auroc']:.3f} "
                f"AUROC, {board.iloc[0]['auroc_missing']:.3f} with one "
                f"modality removed).")
        return (session, ex["scan"], ex["schema"], ex["contract"],
                ex["missing"], note, board, _plot(board.set_index("model")),
                ex["auto_text"], ex["auto_table"])
    except Exception as exc:  # noqa: BLE001
        blank = pd.DataFrame()
        return (session, _err(exc), blank, "", blank, "", blank, None, "",
                blank)


# --------------------------------------------------------------------------
# step 2 and 3: schema -> contract
# --------------------------------------------------------------------------


def build_contract(session: Session, edited: pd.DataFrame, target: str,
                   keep: list[str], positive: str, image_mode: str,
                   embeddings: str, checkpoint: str,
                   progress=gr.Progress()):
    from medtabx.auto import build_frame, describe

    if session.scan is None or session.scan.table is None:
        return session, "Scan a folder with a table first.", None, None
    try:
        progress(0.1, desc="assembling the typed contract")
        table = session.scan.table.copy()
        edited = pd.DataFrame(edited)
        modalities: dict[str, list[str]] = {}
        for _, row in edited.iterrows():
            col, mod = str(row["column"]), str(row["modality"])
            if col == target or mod in ("ignore", "") or mod not in keep:
                continue
            if col in table.columns:
                modalities.setdefault(mod, []).append(col)
        if not modalities:
            return (session, "No column is assigned to a kept modality.",
                    None, None)

        y = table[target]
        if positive.strip():
            label = y.astype(str).str.strip() == positive.strip()
        else:                        # numeric target: split above the median
            num = pd.to_numeric(y, errors="coerce")
            label = num > num.median()
        table = table.assign(__target__=label.astype(float).where(y.notna()))
        table = table[table["__target__"].notna()].reset_index(drop=True)

        images = None
        if image_mode == "Precomputed embeddings (.npz)" and embeddings:
            progress(0.4, desc="loading embeddings")
            z = np.load(embeddings, allow_pickle=True)
            subs, mods = list(z["subjects"]), list(z["modalities"])
            index = {s: i for i, s in enumerate(subs)}
            rows = [index.get(s) for s in table[session.scan.id_column]]
            images = {}
            for j, m in enumerate(mods):
                arr = np.stack([
                    z["tokens"][i, j] if i is not None else
                    np.full(z["tokens"].shape[2:], np.nan) for i in rows])
                images[m] = arr
        elif image_mode == "Encode volumes now (needs a checkpoint)":
            return (session, "Encoding volumes from the interface is not "
                    "enabled in this build: run `scripts/extract_embeddings.py` "
                    "once and load the .npz here. Encoding thousands of "
                    "volumes belongs in a batch job, not in a web request.",
                    None, None)

        progress(0.7, desc="typing the columns")
        data, builder = build_frame(table, "__target__", modalities,
                                    images=images)
        session.data, session.builder = data, builder
        session.meta = describe(data)
        meta = session.meta
        lines = [
            f"**{meta['n']} patients**, {100 * meta['pos_rate']:.0f}% positive",
            f"**{meta['n_modalities']} modalities**: "
            + ", ".join(meta["modalities"]),
            f"value types present: {', '.join(meta['value_types'])}",
            f"flat view: {meta['n_numeric']} numeric + "
            f"{meta['n_categorical']} categorical columns"
            + (f" + {meta['n_image_modalities']} image modalities"
               if meta["n_image_modalities"] else ""),
        ]
        miss = pd.DataFrame(
            [{"modality": m, "missing for": f"{100 * v:.0f}% of patients"}
             for m, v in meta["modality_missing"].items()])
        if meta["n"] < 60:
            lines.append("_Fewer than 60 patients: treat every difference "
                         "below 0.05 AUROC as noise._")
        return session, "\n\n".join(lines), miss, _suggest(meta)
    except Exception as exc:  # noqa: BLE001
        return session, _err(exc), None, None


def _suggest(meta: dict) -> str:
    """What the rule prior would say, before anything is fitted."""
    from medtabx.auto import Objective, prior
    try:
        cands = prior(meta, Objective.preset("balanced"))[:5]
    except Exception:  # noqa: BLE001
        return ""
    lines = ["**Before fitting anything, the rules suggest:**", ""]
    for c in cands:
        lines.append(f"- `{c.model}` — {c.label}")
    if cands and cands[0].reasons:
        lines += ["", "_Why the first one:_"] + [
            f"  - {r}" for r in cands[0].reasons[:3]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# step 4: run
# --------------------------------------------------------------------------


PRESETS = {
    "Quick look (linear + trees, 3 folds)":
        (["lr", "lr@late", "rf", "hgb"], 3, 1),
    "Standard (adds boosting and subsets, 5 folds)":
        (["lr", "lr@late", "rf@late", "hgb", "hgb@late", "et@subset"], 5, 1),
    "Fusion comparison (how modalities are combined, 5 folds)":
        (["lr", "lr@late", "lr@subset", "lr@acq",
          "hgb", "hgb@late", "hgb@acq"], 5, 1),
    "Everything available here (slow)": ("all", 5, 1),
}


def run_compare(session: Session, preset: str, extra: str,
                progress=gr.Progress()):
    from medtabx.auto import compare_models, list_models

    if session.data is None:
        return session, "Build the contract first.", None, None
    try:
        models, folds, repeats = PRESETS[preset]
        if models == "all":
            models = list_models(session.data)
        if extra.strip():
            models = list(dict.fromkeys(
                list(models) + [m.strip() for m in extra.split(",")
                                if m.strip()]))
        progress(0.05, desc=f"fitting {len(models)} models on {folds} folds")
        board = compare_models(session.data, models, n_splits=folds,
                               repeats=repeats, log=None)
        session.rows, session.leaderboard = board.rows, board.table
        table = board.table.reset_index()
        cols = [c for c in ("model", "label", "auroc", "auroc_sd", "auprc",
                            "brier", "ece", "auroc_missing", "missing_drop",
                            "fit_s", "error") if c in table]
        pretty = table[cols].round(3)
        best = board.best
        note = (f"**Best on full inputs: `{best}`** "
                f"({board.table.loc[best, 'auroc']:.3f} AUROC). "
                "The *missing* columns show what happens when one modality is "
                "removed at test time, which is the number to look at before "
                "deploying anywhere.")
        return session, note, pretty, _plot(board.table)
    except Exception as exc:  # noqa: BLE001
        return session, _err(exc), None, None


def run_auto(session: Session, objective: str, budget: int,
             progress=gr.Progress()):
    from medtabx.auto import AutoRecipe

    if session.data is None:
        return session, "Build the contract first.", None, None
    try:
        progress(0.1, desc="rule prior")
        auto = AutoRecipe(objective, budget=int(budget), n_splits=5,
                          repeats=1)
        progress(0.3, desc="successive halving")
        auto.fit(session.data, seed=0)
        session.recipe = {"model": auto.best.model, "label": auto.best.label,
                          "objective": objective, "budget": int(budget),
                          "auroc": float(auto.best.auroc),
                          "reasons": auto.best.reasons,
                          "meta": session.meta}
        table = auto.table[["model", "label", "prior", "rung", "folds",
                            "auroc", "ece", "missing_drop", "fit_s",
                            "utility"]].round(3)
        return session, auto.explain(), table, None
    except Exception as exc:  # noqa: BLE001
        return session, _err(exc), None, None


def _plot(board: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = board[board["auroc"].notna()].sort_values("auroc")
    if ok.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(ok) + 1.2))
    y = np.arange(len(ok))
    ax.barh(y, ok["auroc"], color="#5185C0", height=0.6, label="all inputs")
    if "auroc_missing" in ok:
        ax.scatter(ok["auroc_missing"], y, color="#C96144", s=22, zorder=3,
                   label="one modality missing")
    ax.set_yticks(y)
    ax.set_yticklabels(ok["label"], fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("AUROC")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# step 5: export
# --------------------------------------------------------------------------


def do_export(session: Session, folder: str):
    if session.leaderboard is None and session.recipe is None:
        return "Run a comparison or a recommendation first.", None
    out = Path(folder.strip() or "medtabx_run").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if session.leaderboard is not None:
        session.leaderboard.to_csv(out / "leaderboard.csv")
        session.rows.to_csv(out / "per_fold.csv", index=False)
        written += ["leaderboard.csv", "per_fold.csv"]
    if session.recipe is not None:
        (out / "recipe.json").write_text(json.dumps(session.recipe, indent=2,
                                                    default=float))
        written.append("recipe.json")
    if session.scan is not None:
        (out / "schema.json").write_text(json.dumps(
            {"root": str(session.scan.root),
             "id_column": session.scan.id_column,
             "columns": [asdict(c) for c in session.scan.columns]},
            indent=2))
        written.append("schema.json")
    script = _script(session)
    (out / "reproduce.py").write_text(script)
    written.append("reproduce.py")
    return (f"Wrote {', '.join(written)} to `{out}`.", script)


def _script(session: Session) -> str:
    model = (session.recipe or {}).get("model") or (
        session.leaderboard.index[0] if session.leaderboard is not None
        else "lr@late")
    root = session.scan.root if session.scan else "<study folder>"
    return f'''"""Reproduce this run without the interface."""

from medtabx.app.scan import scan_study
from medtabx.auto import build_frame, compare_models, create_model

scan = scan_study(r"{root}")
table = scan.table
modalities = {{}}                      # edit as in the interface
for c in scan.columns:
    if c.modality not in ("other", "ignore"):
        modalities.setdefault(c.modality, []).append(c.name)

data, builder = build_frame(table, "<target column>", modalities)
board = compare_models(data, ["{model}"], n_splits=5)
print(board)

model = create_model("{model}").fit(data, seed=0)
'''


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="MedTabX Studio") as demo:
        session = gr.State(Session())
        gr.Markdown(INTRO)

        with gr.Row():
            example = gr.Dropdown(
                label="Open a worked example instead (loads stored results, "
                      "computes nothing)",
                choices=demo_data.available(),
                value=(demo_data.available() or [None])[0], scale=4)
            example_btn = gr.Button("Load example", scale=1)

        with gr.Tab("1 · Scan"):
            with gr.Row():
                folder = gr.Textbox(
                    label="Study folder", scale=4,
                    placeholder="/data/my_study  (BIDS-style or one folder "
                                "per subject)")
                scan_btn = gr.Button("Scan folder", variant="primary",
                                     scale=1)
            scan_md = gr.Markdown(STATE_HELP)
            with gr.Row():
                mod_df = gr.Dataframe(label="What was found", interactive=False,
                                      wrap=True)
            gr.Markdown("**Proposed schema.** Edit the *modality* and "
                        "*semantic type* cells if the guess is wrong; set a "
                        "modality to `ignore` to drop a column.")
            col_df = gr.Dataframe(label="Columns", interactive=True, wrap=True)

        with gr.Tab("2 · Target and contract"):
            with gr.Row():
                target = gr.Dropdown(label="Predict which column?",
                                     choices=[], interactive=True)
                positive = gr.Textbox(
                    label="Positive class (blank = split numeric at median)",
                    placeholder="e.g. 1, or 'poor'")
            keep = gr.CheckboxGroup(label="Modalities to include", choices=[])
            with gr.Row():
                image_mode = gr.Radio(
                    label="Imaging",
                    choices=["Ignore images (tables only)",
                             "Precomputed embeddings (.npz)",
                             "Encode volumes now (needs a checkpoint)"],
                    value="Ignore images (tables only)")
                embeddings = gr.Textbox(label="Embeddings .npz",
                                        placeholder="results/embeddings.npz")
                checkpoint = gr.Textbox(label="Encoder checkpoint",
                                        placeholder="data/encoders/...")
            build_btn = gr.Button("Build contract", variant="primary")
            contract_md = gr.Markdown()
            with gr.Row():
                miss_df = gr.Dataframe(label="Missing per modality",
                                       interactive=False)
                suggest_md = gr.Markdown()

        with gr.Tab("3 · Run"):
            gr.Markdown("Every model is fitted on the same folds and scored "
                        "twice: with all inputs, and with one modality "
                        "removed at test time.")
            with gr.Row():
                preset = gr.Dropdown(label="Which models",
                                     choices=list(PRESETS),
                                     value=list(PRESETS)[0])
                extra = gr.Textbox(
                    label="Also run (comma separated ids)",
                    placeholder="cat@late, tabpfn@acq, pf_ft")
            compare_btn = gr.Button("Compare models", variant="primary")
            compare_md = gr.Markdown()
            board_df = gr.Dataframe(label="Leaderboard", interactive=False,
                                    wrap=True)
            board_plot = gr.Plot(label="All inputs vs one modality missing")

            gr.Markdown("---\n### Or let the recommender choose")
            with gr.Row():
                objective = gr.Dropdown(
                    label="What matters", value="balanced",
                    choices=["balanced", "accuracy", "robust", "fast"])
                budget = gr.Slider(label="Fit budget", minimum=10,
                                   maximum=200, step=10, value=40)
            auto_btn = gr.Button("Recommend a pipeline")
            auto_md = gr.Markdown()
            auto_df = gr.Dataframe(label="Candidates", interactive=False,
                                   wrap=True)

        with gr.Tab("4 · Export"):
            out_dir = gr.Textbox(label="Write results to",
                                 value="medtabx_run")
            export_btn = gr.Button("Export", variant="primary")
            export_md = gr.Markdown()
            script_box = gr.Code(label="reproduce.py", language="python")

        with gr.Tab("Help"):
            gr.Markdown(HELP_TEXT)

        scan_btn.click(do_scan, [folder, session],
                       [session, scan_md, mod_df, col_df, target, keep])
        build_btn.click(build_contract,
                        [session, col_df, target, keep, positive, image_mode,
                         embeddings, checkpoint],
                        [session, contract_md, miss_df, suggest_md])
        compare_btn.click(run_compare, [session, preset, extra],
                          [session, compare_md, board_df, board_plot])
        auto_btn.click(run_auto, [session, objective, budget],
                       [session, auto_md, auto_df, gr.State(None)])
        export_btn.click(do_export, [session, out_dir],
                         [export_md, script_box])
        example_btn.click(
            load_example, [example, session],
            [session, scan_md, col_df, contract_md, miss_df, compare_md,
             board_df, board_plot, auto_md, auto_df])
    return demo


HELP_TEXT = """
### What the columns of the leaderboard mean

| Column | Meaning |
|---|---|
| `auroc` | discrimination with every input present |
| `auroc_missing` | mean over scenarios in which one modality is removed at test time |
| `missing_drop` | the difference; a large value means the model leans on one source |
| `brier`, `ece` | calibration: how close predicted probabilities are to observed rates |
| `fit_s` | seconds to fit one fold on this machine |

### Model ids

`base` runs one model on all columns. `base@late` fits one model per
modality and averages those whose modality is present. `base@subset` fits
members on random subsets of modalities and scores a patient only with
members whose modalities are all observed. `base@acq` also fits one model per
modality, but instead of averaging them uniformly it weights each one per
patient by how much that modality predicts on its own (estimated inside the
training fold) and by how ordinary this patient's acquisition looks next to
the training distribution --- a scan with motion or a different protocol is
trusted less for that patient alone. Bases include `lr`, `rf`, `et`,
`hgb`, and, when installed, `lgbm`, `xgb`, `cat`, `tabpfn`, the PyTorch
Frame models (`pf_ft`, `pf_mlp`, ...) and `medtabx`.

### Two habits worth keeping

1. Read `auroc_missing`, not only `auroc`. A model that never saw an absent
   modality during training can look excellent and collapse in the clinic.
2. Compare against the best single modality before believing that combining
   sources helped.

### What this interface does not do

It does not encode raw volumes: run `scripts/extract_embeddings.py` once on
a machine with a GPU and load the resulting `.npz` here. It also does not
replace a statistical analysis plan; it is a tool for a fast, controlled
first look at a dataset.
"""


def _free_port(preferred: int, host: str, tries: int = 20) -> int:
    """First free port at or after ``preferred``.

    A suspended or forgotten instance keeps holding its port, which would
    otherwise stop the interface from starting with a traceback the user
    cannot act on.
    """
    import socket

    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
    return preferred


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="MedTabX Studio")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public link")
    ap.add_argument("--open", action="store_true",
                    help="open the interface in a browser on start")
    args = ap.parse_args(argv)
    port = _free_port(args.port, args.host)
    if port != args.port:
        print(f"port {args.port} is in use (an earlier instance may be "
              f"suspended; 'kill %1' ends it) -- using {port} instead")
    build_interface().launch(server_name=args.host, server_port=port,
                             share=args.share, theme=THEME,
                             inbrowser=args.open, show_error=True)
    return 0
