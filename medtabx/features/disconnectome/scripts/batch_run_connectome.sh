#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_connectome.sh"

usage() {
    cat <<'EOF'
Batch wrapper for run_connectome.sh.

Provide a TSV with header:
subject_id  dwi  bvec  bval  subjects_dir  output_dir  t1_to_dwi_mat  [dwi_mask]  [parcellation_dwi]

Example:
  batch_run_connectome.sh --manifest subjects.tsv --atlas desikan --threads 8
EOF
}

MANIFEST=""
ATLAS="desikan"
THREADS=""
N_STREAMLINES=""
PYTHON_BIN="python3"
MIN_EDGE_WEIGHT="0"
KEEP_WORKDIR="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --atlas) ATLAS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --n-streamlines) N_STREAMLINES="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --min-edge-weight) MIN_EDGE_WEIGHT="$2"; shift 2 ;;
        --keep-workdir) KEEP_WORKDIR="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "${MANIFEST}" ]] || { usage; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "Manifest not found: ${MANIFEST}" >&2; exit 1; }

{
    read -r _
    while IFS=$'\t' read -r subject_id dwi bvec bval subjects_dir output_dir t1_to_dwi_mat dwi_mask parcellation_dwi; do
        [[ -n "${subject_id}" ]] || continue
        cmd=(
            "${RUN_SCRIPT}"
            --subject-id "${subject_id}"
            --dwi "${dwi}"
            --bvec "${bvec}"
            --bval "${bval}"
            --subjects-dir "${subjects_dir}"
            --output-dir "${output_dir}"
            --atlas "${ATLAS}"
            --python "${PYTHON_BIN}"
            --min-edge-weight "${MIN_EDGE_WEIGHT}"
        )
        [[ -n "${THREADS}" ]] && cmd+=(--threads "${THREADS}")
        [[ -n "${N_STREAMLINES}" ]] && cmd+=(--n-streamlines "${N_STREAMLINES}")
        [[ -n "${t1_to_dwi_mat}" ]] && cmd+=(--t1-to-dwi-mat "${t1_to_dwi_mat}")
        [[ -n "${dwi_mask}" ]] && cmd+=(--dwi-mask "${dwi_mask}")
        [[ -n "${parcellation_dwi}" ]] && cmd+=(--parcellation-dwi "${parcellation_dwi}")
        [[ "${KEEP_WORKDIR}" == "1" ]] && cmd+=(--keep-workdir)

        echo "[batch_connectome] ${subject_id}"
        "${cmd[@]}"
    done
} < "${MANIFEST}"
