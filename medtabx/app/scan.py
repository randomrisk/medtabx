"""Scan a study folder and guess a typed schema from it.

The user of the interface points at a directory. This module walks it, finds
the subjects, recognizes which imaging modality each file is, locates the
tabular files, and proposes a schema: which column belongs to which modality
and which semantic type it carries. Everything it returns is a *proposal*
that the interface shows for editing -- nothing here is silently accepted.

Layouts understood out of the box:

* BIDS-like: ``<root>/sub-XXX/ses-YY/<sub>_<ses>_<modality>.nii.gz``
* flat per-subject folders: ``<root>/<subject>/<modality>.nii.gz``
* derivative trees next to the raw tree (``derivatives/``), used for masks

Anything not recognized is reported rather than dropped, so the user can see
what the scan missed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import medtabx.core.stype as st

IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mgz", ".nrrd")
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls")

#: Filename fragment -> (modality name, semantic type). Order matters: the
#: first match wins, so longer and more specific keys come first.
MODALITY_PATTERNS: list[tuple[str, tuple[str, object]]] = [
    ("lesion-msk", ("lesion_mask", st.mask_3d)),
    ("lesion_mask", ("lesion_mask", st.mask_3d)),
    ("lesionmask", ("lesion_mask", st.mask_3d)),
    ("_mask", ("mask", st.mask_3d)),
    ("-msk", ("mask", st.mask_3d)),
    ("seg", ("mask", st.mask_3d)),
    ("dwi_b1000", ("dwi", st.image_3d)),
    ("dwi", ("dwi", st.image_3d)),
    ("adc", ("adc", st.image_3d)),
    ("b0", ("b0", st.image_3d)),
    ("flair", ("flair", st.image_3d)),
    ("swi", ("swi", st.image_3d)),
    ("susc", ("swi", st.image_3d)),
    ("t1", ("t1", st.image_3d)),
    ("t2", ("t2", st.image_3d)),
    ("ncct", ("ncct", st.image_3d)),
    ("cta", ("cta", st.image_3d)),
    ("tmax", ("tmax", st.image_3d)),
    ("cbf", ("cbf", st.image_3d)),
    ("cbv", ("cbv", st.image_3d)),
    ("mtt", ("mtt", st.image_3d)),
    ("perfusion", ("perfusion_map", st.image_3d)),
    ("pet", ("pet", st.image_3d)),
]

SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+|patient[_-]?\d+|case[_-]?\d+)",
                        re.IGNORECASE)

#: Column-name fragment -> (modality group, semantic type) for tabular files.
COLUMN_PATTERNS: list[tuple[str, tuple[str, object]]] = [
    ("age", ("demographics", st.numerical)),
    ("sex", ("demographics", st.categorical)),
    ("gender", ("demographics", st.categorical)),
    ("bmi", ("demographics", st.numerical)),
    ("nihss", ("presentation", st.numerical)),
    ("gcs", ("presentation", st.numerical)),
    ("mrs", ("presentation", st.numerical)),
    ("onset", ("presentation", st.numerical)),
    ("wake", ("presentation", st.categorical)),
    ("glucose", ("labs", st.lab_value)),
    ("leuco", ("labs", st.lab_value)),
    ("crp", ("labs", st.lab_value)),
    ("inr", ("labs", st.lab_value)),
    ("platelet", ("labs", st.lab_value)),
    ("creatinin", ("labs", st.lab_value)),
    ("hba1c", ("labs", st.lab_value)),
    ("cholesterol", ("labs", st.lab_value)),
    ("hypertension", ("history", st.categorical)),
    ("diabetes", ("history", st.categorical)),
    ("fibrillation", ("history", st.categorical)),
    ("hyperlipid", ("history", st.categorical)),
    ("smok", ("history", st.categorical)),
    ("anticoag", ("history", st.medication)),
    ("statin", ("history", st.medication)),
    ("drug", ("history", st.medication)),
    ("icd", ("history", st.medical_code)),
]


@dataclass
class ColumnGuess:
    name: str
    modality: str
    stype: str
    kind: str           # "numeric" | "categorical"
    missing: float
    n_unique: int
    example: str


@dataclass
class ScanResult:
    root: Path
    subjects: list[str] = field(default_factory=list)
    image_modalities: dict[str, dict[str, Path]] = field(default_factory=dict)
    table_files: list[Path] = field(default_factory=list)
    table: pd.DataFrame | None = None
    id_column: str | None = None
    columns: list[ColumnGuess] = field(default_factory=list)
    candidate_targets: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- views the interface renders -------------------------------------
    def modality_table(self) -> pd.DataFrame:
        rows = []
        n = max(len(self.subjects), 1)
        for mod, per_subject in sorted(self.image_modalities.items()):
            rows.append({"modality": mod, "type": "image / mask",
                         "subjects with it": len(per_subject),
                         "coverage": f"{100 * len(per_subject) / n:.0f}%"})
        groups = Counter(c.modality for c in self.columns)
        for mod, k in sorted(groups.items()):
            miss = np.mean([c.missing for c in self.columns
                            if c.modality == mod])
            rows.append({"modality": mod, "type": f"{k} tabular columns",
                         "subjects with it":
                             int(round((1 - miss) * len(self.table)))
                             if self.table is not None else 0,
                         "coverage": f"{100 * (1 - miss):.0f}%"})
        return pd.DataFrame(rows)

    def column_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "column": c.name, "modality": c.modality, "semantic type": c.stype,
            "missing": f"{100 * c.missing:.0f}%", "distinct": c.n_unique,
            "example": c.example} for c in self.columns])

    def summary(self) -> str:
        lines = [f"**{len(self.subjects)} subjects** under `{self.root}`"]
        if self.image_modalities:
            lines.append(f"**{len(self.image_modalities)} imaging "
                         f"modalities**: " +
                         ", ".join(sorted(self.image_modalities)))
        if self.table is not None:
            n_files = len(self.table_files)
            src = (f"{n_files} per-subject tables" if n_files > 3
                   else ", ".join(f"`{f.name}`" for f in self.table_files))
            lines.append(f"**{len(self.columns)} tabular columns** and "
                         f"{len(self.table)} rows from {src}")
            if self.id_column:
                lines.append(f"subject identifier: `{self.id_column}`")
        for note in self.notes:
            lines.append(f"- {note}")
        if self.unmatched:
            lines.append(f"- {len(self.unmatched)} file patterns were not "
                         f"recognized (e.g. `{self.unmatched[0]}`); assign "
                         "them manually if they matter")
        return "\n\n".join(lines)


def _match_modality(path: Path) -> tuple[str, object] | None:
    name = path.name.lower()
    for key, value in MODALITY_PATTERNS:
        if key in name:
            return value
    return None


def _subject_of(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts:
        m = SUBJECT_RE.search(part)
        if m:
            return m.group(0)
    return None


def _guess_columns(df: pd.DataFrame, id_column: str | None,
                   max_levels: int = 10) -> list[ColumnGuess]:
    out = []
    for col in df.columns:
        if col == id_column:
            continue
        s = df[col]
        low = str(col).lower()
        modality, stype = "other", None
        for key, value in COLUMN_PATTERNS:
            if key in low:
                modality, stype = value
                break
        nun = int(s.nunique(dropna=True))
        if s.dtype == object or s.dtype == bool or nun <= max_levels:
            kind = "categorical"
        else:
            kind = "numeric"
        if stype is None:
            stype = st.categorical if kind == "categorical" else st.numerical
        example = "" if s.dropna().empty else str(s.dropna().iloc[0])[:18]
        out.append(ColumnGuess(
            name=str(col), modality=modality, stype=str(stype).split(".")[-1],
            kind=kind, missing=float(s.isna().mean()), n_unique=nun,
            example=example))
    return out


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    except Exception:  # noqa: BLE001 - a bad table must not stop the scan
        return None



def _assemble_table(files: list[Path], root: Path,
                    subjects: list[str]) -> tuple[pd.DataFrame | None,
                                                  list[Path]]:
    """Return one cohort table.

    Two layouts are common. A study may keep a single table with one row per
    subject, or---as in BIDS-style releases---one small table per subject and
    per session. In the second case the tables are concatenated, the subject
    identifier is taken from the path, and groups that describe different
    aspects of the same subject (baseline, outcome) are merged on it.
    """
    groups: dict[tuple, list[tuple[Path, pd.DataFrame]]] = defaultdict(list)
    for f in files:
        df = _read_table(f)
        if df is None or df.empty:
            continue
        groups[tuple(map(str, df.columns))].append((f, df))

    assembled: list[pd.DataFrame] = []
    used: list[Path] = []
    single: tuple[pd.DataFrame, list[Path]] | None = None
    for cols, items in groups.items():
        rows = sum(len(df) for _, df in items)
        if len(items) >= 5 and rows / len(items) <= 3:
            # one table per subject: stack them and recover the identifier
            frames = []
            for f, df in items:
                sid = _subject_of(f, root) or f.stem
                df = df.copy()
                df.insert(0, "subject_id", sid)
                frames.append(df)
            assembled.append(pd.concat(frames, ignore_index=True)
                             .groupby("subject_id", as_index=False).first())
            used.extend(f for f, _ in items)
        elif single is None or len(cols) > len(single[0].columns):
            single = (items[0][1], [items[0][0]])

    if assembled:
        table = assembled[0]
        for other in assembled[1:]:
            table = table.merge(other, on="subject_id", how="outer",
                                suffixes=("", "_dup"))
        table = table.loc[:, ~table.columns.str.endswith("_dup")]
        if subjects:
            table = table[table.subject_id.isin(subjects)]
        return table.reset_index(drop=True), used
    if single is not None:
        return single
    return None, []


def scan_study(root: str | Path, max_files: int = 200_000) -> ScanResult:
    """Walk ``root`` and propose a schema."""
    root = Path(root).expanduser().resolve()
    res = ScanResult(root=root)
    if not root.exists():
        res.notes.append(f"{root} does not exist")
        return res

    images: dict[str, dict[str, Path]] = defaultdict(dict)
    subjects: set[str] = set()
    unmatched: Counter = Counter()
    seen = 0
    for path in root.rglob("*"):
        seen += 1
        if seen > max_files:
            res.notes.append(f"stopped after {max_files} files")
            break
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(TABLE_SUFFIXES):
            res.table_files.append(path)
            continue
        if not name.endswith(IMAGE_SUFFIXES):
            continue
        subject = _subject_of(path, root)
        if subject:
            subjects.add(subject)
        hit = _match_modality(path)
        if hit is None:
            unmatched[re.sub(r"[0-9]+", "#", path.name)] += 1
            continue
        modality, _ = hit
        if subject:
            images[modality].setdefault(subject, path)

    res.subjects = sorted(subjects)
    res.image_modalities = {k: dict(v) for k, v in images.items()}
    res.unmatched = [f"{k} ({v})" for k, v in unmatched.most_common(5)]

    res.table, used = _assemble_table(res.table_files, root, res.subjects)
    if res.table is not None:
        res.table_files = used
        df = res.table
        for col in df.columns:
            values = df[col].astype(str)
            if str(col) == "subject_id":
                res.id_column = "subject_id"
                break
            if values.str.contains("sub-", case=False).mean() > 0.5 or \
                    str(col).lower() in ("subject_id", "subject", "id",
                                         "patient_id", "case"):
                res.id_column = str(col)
                break
        res.columns = _guess_columns(df, res.id_column)
        res.candidate_targets = [
            c.name for c in res.columns
            if c.kind == "categorical" and c.n_unique <= 6
        ] + [c.name for c in res.columns
             if c.kind == "numeric" and c.n_unique <= 8]
        if not res.id_column:
            res.notes.append(
                "no subject identifier column was recognized; pick one "
                "before running, otherwise images cannot be joined to rows")
    elif res.image_modalities:
        res.notes.append("no tabular file was found: the run will use "
                         "imaging modalities only")

    if res.image_modalities and res.subjects:
        complete = sum(
            1 for s in res.subjects
            if all(s in per for per in res.image_modalities.values()))
        res.notes.append(
            f"{complete} of {len(res.subjects)} subjects have every imaging "
            "modality; the rest are kept, with the absent modality masked")
    return res
