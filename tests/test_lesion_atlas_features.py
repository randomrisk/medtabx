"""Tests for scripts/lesion_atlas_features.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / \
    "lesion_atlas_features.py"


@pytest.fixture(scope="module")
def laf():
    spec = importlib.util.spec_from_file_location("lesion_atlas_features",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules, so the
    # module has to be registered before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def atlas_dir(laf):
    directory = laf.default_atlas_dir()
    if not directory.exists():
        pytest.skip("vendored atlas pack not present")
    return directory


@pytest.fixture(scope="module")
def arterial(laf, atlas_dir):
    return laf.load_atlas("arterial", atlas_dir)


# Core arithmetic ############################################################


def test_lesion_load_counts_voxels_per_label(laf):
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[:2] = 1
    labels[2:] = 2
    atlas = laf.Atlas("toy", labels, ["a", "b"],
                      np.array([32.0, 32.0]))

    lesion = np.zeros_like(labels, dtype=np.uint8)
    lesion[0, :, :] = 1  # 16 voxels, all in label 1
    lesion[3, 0, 0] = 1  # 1 voxel in label 2

    assert laf.lesion_load(lesion, atlas).tolist() == [16, 1]


def test_lesion_load_ignores_voxels_outside_the_atlas(laf):
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[0] = 1
    atlas = laf.Atlas("toy", labels, ["a"], np.array([16.0]))

    lesion = np.ones_like(labels, dtype=np.uint8)  # also covers label 0
    assert laf.lesion_load(lesion, atlas).tolist() == [16]


def test_qfv_is_count_over_region_volume(laf):
    counts = np.array([5, 0, 10])
    volumes = np.array([10.0, 20.0, 40.0])
    assert np.allclose(laf.quantified_fractional_volume(counts, volumes),
                       [0.5, 0.0, 0.25])


def test_qfv_zero_volume_region_is_zero_not_nan(laf):
    qfv = laf.quantified_fractional_volume(np.array([3, 0]),
                                           np.array([0.0, 0.0]))
    assert qfv.tolist() == [0.0, 0.0]


def test_shape_mismatch_names_the_fix(laf, arterial):
    with pytest.raises(ValueError, match="Register the mask into MNI152"):
        laf.lesion_load(np.zeros((10, 10, 10), dtype=np.uint8), arterial)


# Atlases ####################################################################


@pytest.mark.parametrize("name, regions", [
    ("arterial", 32),
    ("lobe", 31),
    ("aspects", 21),
    ("aspectpc", 8),
    ("ventricles", 20),
])
def test_requested_atlases_load_on_the_mni_grid(laf, atlas_dir, name,
                                                regions):
    atlas = laf.load_atlas(name, atlas_dir)
    assert atlas.num_regions == regions
    assert atlas.labels.shape == laf.MNI_SHAPE
    assert len(atlas.roi_volumes) == regions
    assert (atlas.roi_volumes > 0).all()


def test_volume_table_is_read_as_a_row_not_a_column(arterial):
    # Regression for the OpenADS bug: `.iloc[:, 0]` on the one-row lookup
    # table yields a single number (the first region's volume), which NumPy
    # then broadcasts over every region.
    assert len(arterial.roi_volumes) == arterial.num_regions
    assert len(set(arterial.roi_volumes.tolist())) > 1


def test_region_names_come_from_the_lookup_table(arterial):
    assert arterial.roi_names[:4] == ["ACAL", "ACAR", "MLSL", "MLSR"]


def test_label_count_must_match_the_volume_table(laf, tmp_path):
    labels = np.zeros(laf.MNI_SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 1
    labels[0, 0, 1] = 2
    nib.save(nib.Nifti1Image(labels, np.eye(4)),
             tmp_path / laf.ATLASES["arterial"].filename)
    # Three columns for a two-label atlas.
    pd.DataFrame([[1.0, 1.0, 1.0]], columns=["a", "b", "c"]).to_pickle(
        tmp_path / laf.ATLASES["arterial"].volume_table)

    with pytest.raises(ValueError, match="must correspond one to one"):
        laf.load_atlas("arterial", tmp_path)


def test_unknown_atlas_lists_the_choices(laf, atlas_dir):
    with pytest.raises(ValueError, match="Available"):
        laf.load_atlas("not_an_atlas", atlas_dir)


# End-to-end properties on the real atlases ##################################


def test_a_fully_lesioned_region_has_qfv_exactly_one(laf, arterial):
    # The lookup tables hold each region's exact voxel count, so painting a
    # whole region must give QFV 1.0 for it and 0 everywhere else. With the
    # buggy column read, only label 1 would come out right.
    target = arterial.roi_names.index("MCAIL")
    lesion = (arterial.labels == target + 1).astype(np.uint8)

    row = laf.subject_features(lesion, [arterial], subject_id="s")

    assert row["arterial_qfv_MCAIL"] == pytest.approx(1.0)
    others = [v for k, v in row.items()
              if k.startswith("arterial_qfv_") and k != "arterial_qfv_MCAIL"]
    assert max(others) == 0.0
    assert row["arterial_regions_involved"] == 1


def test_every_region_scores_one_when_the_whole_atlas_is_lesioned(
        laf, arterial):
    lesion = (arterial.labels > 0).astype(np.uint8)
    row = laf.subject_features(lesion, [arterial])
    qfv = [v for k, v in row.items() if k.startswith("arterial_qfv_")]
    assert np.allclose(qfv, 1.0)


def test_hemisphere_split_follows_ras(laf, arterial):
    # In RAS the first axis runs left -> right, so the low half is left.
    lesion = np.zeros(laf.MNI_SHAPE, dtype=np.uint8)
    lesion[10:20, 100:110, 80:90] = 1
    metrics = laf.global_lesion_metrics(lesion, (1.0, 1.0, 1.0))

    assert metrics["lesion_volume_mni_ml"] == pytest.approx(1.0)
    assert metrics["lesion_volume_mni_left_ml"] == pytest.approx(1.0)
    assert metrics["lesion_volume_mni_right_ml"] == 0.0


def test_left_arterial_regions_are_on_the_left(laf, arterial):
    # Guards the atlas orientation: the atlas file is LAS, the arrays are
    # canonicalized to RAS, and "L" regions must land in the low-x half.
    midline = laf.MNI_SHAPE[0] // 2
    for name in ("ACAL", "MCAFL", "PCATL"):
        xs = np.nonzero(arterial.labels == arterial.roi_names.index(name) +
                        1)[0]
        assert np.median(xs) < midline, name


def test_empty_lesion_scores_zero_everywhere(laf, arterial):
    row = laf.subject_features(np.zeros(laf.MNI_SHAPE, dtype=np.uint8),
                               [arterial])
    assert row["lesion_volume_mni_ml"] == 0.0
    assert row["arterial_regions_involved"] == 0
    assert all(v == 0 for k, v in row.items() if "_qfv_" in k)


def test_no_load_emits_only_fractions(laf, arterial):
    lesion = (arterial.labels == 1).astype(np.uint8)
    row = laf.subject_features(lesion, [arterial], include_load=False)
    assert not any("_load_" in k for k in row)
    assert any("_qfv_" in k for k in row)


# CLI ########################################################################


def test_cli_already_mni_writes_one_row_per_subject(laf, arterial, tmp_path):
    lesion = (arterial.labels == 1).astype(np.uint8)
    # Save in RAS so the canonical reload is a no-op and the grid matches.
    path = tmp_path / "mask_mni.nii.gz"
    nib.save(nib.Nifti1Image(lesion, np.diag([1.0, 1.0, 1.0, 1.0])), path)
    out = tmp_path / "features.csv"

    code = laf.main(["--lesion", str(path), "--already-mni", "--subject-id",
                     "sub-1", "--atlases", "arterial", "lobe", "--out",
                     str(out)])
    assert code == 0

    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.loc[0, "subject_id"] == "sub-1"
    assert frame.loc[0, "arterial_qfv_ACAL"] == pytest.approx(1.0)
    assert any(c.startswith("lobe_qfv_") for c in frame.columns)


def test_cli_batch_mode_scores_each_subject_directory(laf, arterial,
                                                      tmp_path):
    root = tmp_path / "cohort"
    for i, label in enumerate((1, 2)):
        subject = root / f"sub-{i}"
        subject.mkdir(parents=True)
        mask = (arterial.labels == label).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, np.eye(4)),
                 subject / f"sub-{i}_stroke_mask.nii.gz")
    out = tmp_path / "cohort.csv"

    code = laf.main(["--batch-root", str(root), "--already-mni",
                     "--atlases", "arterial", "--out", str(out)])
    assert code == 0

    frame = pd.read_csv(out).set_index("subject_id")
    assert list(frame.index) == ["sub-0", "sub-1"]
    assert frame.loc["sub-0", "arterial_qfv_ACAL"] == pytest.approx(1.0)
    assert frame.loc["sub-1", "arterial_qfv_ACAR"] == pytest.approx(1.0)


def test_cli_already_mni_rejects_an_off_grid_mask(laf, tmp_path, capsys):
    path = tmp_path / "wrong.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4)),
             path)
    code = laf.main(["--lesion", str(path), "--already-mni", "--atlases",
                     "arterial", "--out", str(tmp_path / "x.csv")])
    # One subject, and it failed: the run reports failure.
    assert code == 1
    assert "atlas grid" in capsys.readouterr().err


def test_cli_requires_out(laf, capsys):
    assert laf.main(["--lesion", "x.nii.gz", "--already-mni"]) == 1
    assert "--out is required" in capsys.readouterr().err


def test_cli_lists_atlases_without_out(laf, capsys):
    assert laf.main(["--list-atlases"]) == 0
    listing = capsys.readouterr().out
    for name in ("arterial", "lobe", "aspects", "aspectpc", "ventricles"):
        assert name in listing


# Registration (slow; needs antspyx) #########################################


@pytest.mark.slow
def test_registration_places_the_ads_stroke_in_the_left_mca(laf, atlas_dir,
                                                            tmp_path):
    pytest.importorskip("ants")
    subject = (Path(__file__).resolve().parents[1] / "examples" / "data" /
               "ads_case" / "sub-02e8eb42")
    if not subject.exists():
        pytest.skip("ADS fixture not present")

    atlases = [laf.load_atlas(n, atlas_dir)
               for n in ("arterial", "lobe", "aspects")]
    row = laf.process_subject(
        subject_id="sub-02e8eb42",
        lesion_path=subject / "masks" / "sub-02e8eb42_stroke_mask.nii.gz",
        moving_path=subject / "imaging" / "sub-02e8eb42_DWI.nii.gz",
        atlases=atlases, atlas_dir=atlas_dir)

    # The radiology report: left hemisphere, MCA territory, basal ganglia.
    assert row["lesion_volume_mni_left_ml"] > 50 * row[
        "lesion_volume_mni_right_ml"]
    assert row["arterial_qfv_MCAIL"] > 0.5
    assert row["arterial_qfv_MCAIR"] < 0.01
    assert row["lobe_qfv_BasalGanglia_L"] > 0.5
    # Native volume is the physical size and does not depend on registration.
    assert row["lesion_volume_native_ml"] == pytest.approx(114.17, abs=0.1)
