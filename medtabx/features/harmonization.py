"""Subject-level spatial harmonization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .perfusion import resample_to_reference


def load_subject_manifest(subject_root: str | Path) -> dict[str, Any]:
    subject_root = Path(subject_root)
    manifest_path = subject_root / "manifest.json"
    return json.loads(manifest_path.read_text())


def _artifact_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {artifact["path"]: artifact for artifact in manifest.get("artifacts", [])}


def _relative_to_subject(subject_root: Path, path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(subject_root.resolve()))


def harmonize_subject_to_reference(
    subject_root: str | Path,
    reference_relpath: str,
    *,
    include_kinds: set[str] | None = None,
    include_paths: list[str] | None = None,
    output_dir: str | Path | None = None,
    artifact_transforms: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Resample subject artifacts into a declared reference image space.

    This helper does not run registration. It assumes either:
    - the moving artifact already lives in the same physical space and only
      needs grid harmonization, or
    - a known affine transform is provided via ``artifact_transforms``.

    Args:
        subject_root: Root subject directory containing ``manifest.json``.
        reference_relpath: Artifact path relative to ``subject_root`` used as
            the target reference space.
        include_kinds: Artifact kinds to harmonize. Defaults to masks and
            derived images/maps.
        include_paths: Optional explicit artifact relative paths to harmonize.
        output_dir: Optional destination folder. Defaults to
            ``subject_root/harmonized/<reference-stem>``.
        artifact_transforms: Optional mapping from artifact relative path to a
            known affine transform.
    """
    subject_root = Path(subject_root)
    manifest = load_subject_manifest(subject_root)
    artifact_by_path = _artifact_lookup(manifest)
    if reference_relpath not in artifact_by_path:
        raise ValueError(f"Reference artifact '{reference_relpath}' not found in manifest.")

    default_kinds = {"mask", "derived_image"}
    include_kinds = default_kinds if include_kinds is None else include_kinds
    artifact_transforms = {} if artifact_transforms is None else artifact_transforms

    reference_path = subject_root / reference_relpath
    reference_stem = Path(reference_relpath).stem.replace(".nii", "")
    base_output_dir = (
        subject_root / "harmonized" / reference_stem
        if output_dir is None
        else Path(output_dir)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for relpath, artifact in artifact_by_path.items():
        if relpath == reference_relpath:
            continue
        if include_paths is not None and relpath not in include_paths:
            continue
        if include_paths is None and artifact.get("kind") not in include_kinds:
            continue

        moving_path = subject_root / relpath
        if moving_path.suffixes[-2:] != [".nii", ".gz"] and moving_path.suffix != ".nii":
            continue

        kind = "mask" if artifact.get("kind") == "mask" else "image"
        resampled_img = resample_to_reference(
            str(moving_path),
            str(reference_path),
            kind=kind,
            transform=artifact_transforms.get(relpath),
            return_image=True,
        )

        out_path = base_output_dir / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)

        import nibabel as nib

        nib.save(resampled_img, str(out_path))
        outputs[relpath] = str(out_path)

    harmonized_manifest = {
        "subject_id": manifest.get("subject_id"),
        "reference_relpath": reference_relpath,
        "outputs": outputs,
    }
    (base_output_dir / "harmonized_manifest.json").write_text(json.dumps(harmonized_manifest, indent=2))
    return outputs


def build_harmonized_subject_view(
    subject_root: str | Path,
    *,
    reference_relpath: str,
    path_columns: dict[str, str],
    include_columns: list[str] | None = None,
    output_dir: str | Path | None = None,
    artifact_transforms: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Create a subject-level path view with selected columns harmonized.

    Args:
        subject_root: Subject directory containing ``manifest.json``.
        reference_relpath: Relative path used as the harmonization reference.
        path_columns: Mapping from dataset column name to current artifact path.
        include_columns: Optional subset of columns to harmonize. By default,
            all path columns except the reference itself are eligible.
        output_dir: Optional harmonized output root.
        artifact_transforms: Optional artifact-relative-path -> affine mapping.
    """
    subject_root = Path(subject_root)
    include_columns = list(path_columns.keys()) if include_columns is None else include_columns

    include_relpaths: list[str] = []
    for col in include_columns:
        relpath = _relative_to_subject(subject_root, path_columns[col])
        if relpath != reference_relpath:
            include_relpaths.append(relpath)

    outputs = harmonize_subject_to_reference(
        subject_root,
        reference_relpath,
        include_paths=include_relpaths,
        output_dir=output_dir,
        artifact_transforms=artifact_transforms,
    )

    view = dict(path_columns)
    for col in include_columns:
        relpath = _relative_to_subject(subject_root, path_columns[col])
        if relpath in outputs:
            view[col] = outputs[relpath]
    return view
