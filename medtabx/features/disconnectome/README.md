# Connectome Pipeline

This directory contains a lightweight structural connectome pipeline built around
MRtrix3 and FreeSurfer outputs.

## Inputs

- Preprocessed DWI volume
- Matching `bvec` and `bval`
- FreeSurfer `SUBJECTS_DIR` and subject id
- A T1-to-DWI transform, unless the atlas parcellation is already in DWI space

## Main entrypoints

- `scripts/run_connectome.sh`
- `scripts/batch_run_connectome.sh`
- `pipeline.py` for post-processing a raw connectivity matrix into downstream graph artifacts

## Outputs

- `raw/<prefix>_connectome.csv`: raw MRtrix connectome matrix
- `raw/<prefix>_assignments.csv`: streamline-to-node assignments
- `graph/<prefix>_matrix.tsv`: labeled adjacency matrix
- `graph/<prefix>_edges.csv`: edge list for graph tooling
- `graph/<prefix>_node_strengths.csv`: node-level weighted degree summary
- `graph/<prefix>_summary.json`: scalar features for downstream tabular models

## Example

```bash
medtabx/features/disconnectome/scripts/run_connectome.sh \
  --dwi /data/sub-001/dwi_preproc.nii.gz \
  --bvec /data/sub-001/dwi_preproc.bvec \
  --bval /data/sub-001/dwi_preproc.bval \
  --subjects-dir /data/freesurfer \
  --subject-id sub-001 \
  --t1-to-dwi-mat /data/sub-001/t1_to_dwi.txt \
  --output-dir /data/sub-001/connectome_out
```
