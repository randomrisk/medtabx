#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

usage() {
    cat <<'EOF'
Run a structural connectome pipeline from preprocessed DWI + FreeSurfer outputs.

Required:
  --dwi PATH                  Preprocessed DWI NIfTI(.gz)
  --bvec PATH                 FSL bvec file for the DWI
  --bval PATH                 FSL bval file for the DWI
  --subjects-dir PATH         FreeSurfer SUBJECTS_DIR
  --subject-id ID             FreeSurfer subject id
  --output-dir PATH           Output directory

Required unless the parcellation is already in DWI space:
  --t1-to-dwi-mat PATH        Linear transform mapping T1/FreeSurfer space to DWI space

Optional:
  --atlas NAME                desikan (default) or destrieux
  --dwi-mask PATH             Precomputed DWI mask. If omitted, dwi2mask is used.
  --parcellation-dwi PATH     Atlas parcellation already resampled into DWI space
  --fs-color-lut PATH         FreeSurferColorLUT.txt override
  --mrtrix-lut PATH           MRtrix labelconvert LUT override
  --n-streamlines N           Number of streamlines (default: 1000000)
  --threads N                 MRtrix thread count
  --python BIN                Python executable for graph export (default: python3)
  --prefix STR                Output prefix (default: <subject-id>_<atlas>)
  --min-edge-weight FLOAT     Threshold for exported edge CSV (default: 0)
  --keep-workdir              Keep intermediate work directory

Outputs:
  <output-dir>/raw/<prefix>_connectome.csv
  <output-dir>/graph/<prefix>_matrix.tsv
  <output-dir>/graph/<prefix>_edges.csv
  <output-dir>/graph/<prefix>_node_strengths.csv
  <output-dir>/graph/<prefix>_summary.json
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

resolve_fs_color_lut() {
    if [[ -n "${FS_COLOR_LUT:-}" ]]; then
        printf '%s\n' "${FS_COLOR_LUT}"
        return
    fi
    if [[ -n "${FREESURFER_HOME:-}" && -f "${FREESURFER_HOME}/FreeSurferColorLUT.txt" ]]; then
        printf '%s\n' "${FREESURFER_HOME}/FreeSurferColorLUT.txt"
        return
    fi
    echo "Unable to locate FreeSurferColorLUT.txt. Pass --fs-color-lut or set FREESURFER_HOME." >&2
    exit 1
}

resolve_mrtrix_lut() {
    local lut_name="$1"
    if [[ -n "${MRTRIX_LUT:-}" ]]; then
        printf '%s\n' "${MRTRIX_LUT}"
        return
    fi

    local candidates=()
    if command -v tck2connectome >/dev/null 2>&1; then
        local mrbin
        mrbin="$(cd "$(dirname "$(command -v tck2connectome)")/.." && pwd)"
        candidates+=(
            "${mrbin}/share/mrtrix3/labelconvert/${lut_name}"
            "${mrbin}/share/mrtrix3/labelconvert/fs_default.txt"
        )
    fi
    candidates+=(
        "/usr/local/share/mrtrix3/labelconvert/${lut_name}"
        "/usr/share/mrtrix3/labelconvert/${lut_name}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "Unable to locate MRtrix LUT ${lut_name}. Pass --mrtrix-lut." >&2
    exit 1
}

DWI=""
BVEC=""
BVAL=""
SUBJECTS_DIR=""
SUBJECT_ID=""
OUTPUT_DIR=""
ATLAS="desikan"
DWI_MASK=""
PARCELLATION_DWI=""
T1_TO_DWI_MAT=""
FS_COLOR_LUT=""
MRTRIX_LUT=""
N_STREAMLINES="1000000"
THREADS=""
PYTHON_BIN="python3"
PREFIX=""
MIN_EDGE_WEIGHT="0"
KEEP_WORKDIR="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dwi) DWI="$2"; shift 2 ;;
        --bvec) BVEC="$2"; shift 2 ;;
        --bval) BVAL="$2"; shift 2 ;;
        --subjects-dir) SUBJECTS_DIR="$2"; shift 2 ;;
        --subject-id) SUBJECT_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --atlas) ATLAS="$2"; shift 2 ;;
        --dwi-mask) DWI_MASK="$2"; shift 2 ;;
        --parcellation-dwi) PARCELLATION_DWI="$2"; shift 2 ;;
        --t1-to-dwi-mat) T1_TO_DWI_MAT="$2"; shift 2 ;;
        --fs-color-lut) FS_COLOR_LUT="$2"; shift 2 ;;
        --mrtrix-lut) MRTRIX_LUT="$2"; shift 2 ;;
        --n-streamlines) N_STREAMLINES="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --min-edge-weight) MIN_EDGE_WEIGHT="$2"; shift 2 ;;
        --keep-workdir) KEEP_WORKDIR="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "${DWI}" && -n "${BVEC}" && -n "${BVAL}" && -n "${SUBJECTS_DIR}" && -n "${SUBJECT_ID}" && -n "${OUTPUT_DIR}" ]] || {
    usage
    exit 1
}

if [[ -z "${PARCELLATION_DWI}" && -z "${T1_TO_DWI_MAT}" ]]; then
    echo "Provide --t1-to-dwi-mat or --parcellation-dwi." >&2
    exit 1
fi

case "${ATLAS}" in
    desikan)
        APARC_NAME="aparc+aseg.mgz"
        MRTRIX_LUT_NAME="fs_default.txt"
        ;;
    destrieux)
        APARC_NAME="aparc.a2009s+aseg.mgz"
        MRTRIX_LUT_NAME="fs_a2009s.txt"
        ;;
    *)
        echo "Unsupported atlas: ${ATLAS}. Use desikan or destrieux." >&2
        exit 1
        ;;
esac

require_cmd mrconvert
require_cmd dwi2mask
require_cmd dwi2response
require_cmd dwi2fod
require_cmd tckgen
require_cmd tck2connectome
require_cmd labelconvert
require_cmd mrtransform
require_cmd mri_convert
require_cmd "${PYTHON_BIN}"

if [[ -n "${THREADS}" ]]; then
    export MRTRIX_NTHREADS="${THREADS}"
fi

PREFIX="${PREFIX:-${SUBJECT_ID}_${ATLAS}}"
SUBJECT_DIR="${SUBJECTS_DIR}/${SUBJECT_ID}"
APARC_MGZ="${SUBJECT_DIR}/mri/${APARC_NAME}"
RAW_DIR="${OUTPUT_DIR}/raw"
GRAPH_DIR="${OUTPUT_DIR}/graph"
WORK_DIR="${OUTPUT_DIR}/work_${PREFIX}"
mkdir -p "${RAW_DIR}" "${GRAPH_DIR}" "${WORK_DIR}"

if [[ ! -f "${APARC_MGZ}" && -z "${PARCELLATION_DWI}" ]]; then
    echo "Missing FreeSurfer parcellation: ${APARC_MGZ}" >&2
    exit 1
fi

cleanup() {
    if [[ "${KEEP_WORKDIR}" != "1" ]]; then
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT

echo "[connectome] converting DWI to MRtrix format"
mrconvert "${DWI}" "${WORK_DIR}/dwi.mif" -fslgrad "${BVEC}" "${BVAL}" -force

if [[ -n "${DWI_MASK}" ]]; then
    mrconvert "${DWI_MASK}" "${WORK_DIR}/dwi_mask.mif" -force
else
    echo "[connectome] generating DWI mask"
    dwi2mask "${WORK_DIR}/dwi.mif" "${WORK_DIR}/dwi_mask.mif" -force
fi

echo "[connectome] estimating response function"
dwi2response tournier \
    "${WORK_DIR}/dwi.mif" \
    "${WORK_DIR}/response_wm.txt" \
    -mask "${WORK_DIR}/dwi_mask.mif" \
    -force

echo "[connectome] computing FOD image"
dwi2fod csd \
    "${WORK_DIR}/dwi.mif" \
    "${WORK_DIR}/response_wm.txt" \
    "${WORK_DIR}/fod_wm.mif" \
    -mask "${WORK_DIR}/dwi_mask.mif" \
    -force

if [[ -n "${PARCELLATION_DWI}" ]]; then
    echo "[connectome] using provided DWI-space parcellation"
    mrconvert "${PARCELLATION_DWI}" "${WORK_DIR}/nodes_dwi.mif" -datatype uint32 -force
else
    FS_COLOR_LUT="$(resolve_fs_color_lut)"
    MRTRIX_LUT="$(resolve_mrtrix_lut "${MRTRIX_LUT_NAME}")"

    echo "[connectome] converting FreeSurfer parcellation to DWI space"
    mri_convert "${APARC_MGZ}" "${WORK_DIR}/aparc_raw.nii.gz" >/dev/null
    labelconvert \
        "${WORK_DIR}/aparc_raw.nii.gz" \
        "${FS_COLOR_LUT}" \
        "${MRTRIX_LUT}" \
        "${WORK_DIR}/nodes_t1.mif" \
        -force
    mrtransform \
        "${WORK_DIR}/nodes_t1.mif" \
        "${WORK_DIR}/nodes_dwi.mif" \
        -linear "${T1_TO_DWI_MAT}" \
        -template "${WORK_DIR}/dwi.mif" \
        -interp nearest \
        -datatype uint32 \
        -force

    awk 'NF >= 2 && $1 !~ /^#/ {print $2}' "${MRTRIX_LUT}" > "${WORK_DIR}/node_labels.txt"
fi

echo "[connectome] running probabilistic tractography"
tckgen \
    "${WORK_DIR}/fod_wm.mif" \
    "${WORK_DIR}/tracks.tck" \
    -algorithm iFOD2 \
    -seed_image "${WORK_DIR}/dwi_mask.mif" \
    -mask "${WORK_DIR}/dwi_mask.mif" \
    -select "${N_STREAMLINES}" \
    -cutoff 0.1 \
    -minlength 20 \
    -step 0.2 \
    -force

RAW_MATRIX="${RAW_DIR}/${PREFIX}_connectome.csv"
ASSIGNMENTS="${RAW_DIR}/${PREFIX}_assignments.csv"
echo "[connectome] building connectivity matrix"
tck2connectome \
    "${WORK_DIR}/tracks.tck" \
    "${WORK_DIR}/nodes_dwi.mif" \
    "${RAW_MATRIX}" \
    -symmetric \
    -zero_diagonal \
    -out_assignments "${ASSIGNMENTS}" \
    -force

echo "[connectome] exporting downstream graph artifacts"
export_cmd=(
    "${PYTHON_BIN}" -m medtabx.features.disconnectome.pipeline
    --matrix "${RAW_MATRIX}"
    --output-dir "${GRAPH_DIR}"
    --prefix "${PREFIX}"
    --min-weight "${MIN_EDGE_WEIGHT}"
)

if [[ -f "${WORK_DIR}/node_labels.txt" ]]; then
    cp "${WORK_DIR}/node_labels.txt" "${GRAPH_DIR}/${PREFIX}_labels.txt"
    export_cmd+=(--labels-file "${WORK_DIR}/node_labels.txt")
fi

"${export_cmd[@]}"

echo "[connectome] done"
echo "  raw matrix: ${RAW_MATRIX}"
echo "  graph dir : ${GRAPH_DIR}"
