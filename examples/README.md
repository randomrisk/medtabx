# Examples

| Script | What it shows | Data |
|---|---|---|
| `auto_recipe_demo.py` | Typed contract, model comparison and `AutoRecipe` | Synthetic, built in the script |
| `train_and_evaluate.py` | Training, evaluation and prediction with the MedTabX engine | Synthetic, built in the script |
| `end_to_end_pipeline.py` | Loading and encoding tabular, imaging and vital-sign inputs | Sample files (see below) |
| `extract_features.py` | Feature extraction for each supported input type | Sample files (see below) |

Run from the repository root, for example `python examples/auto_recipe_demo.py`.
`auto_recipe_demo.py` takes about 5 minutes on a 4-core CPU with the core
install; `train_and_evaluate.py` takes less than 1 minute.

## Sample files

`end_to_end_pipeline.py` and `extract_features.py` use small public sample
files. Download them once:

```bash
python examples/data/download_samples.py
python examples/data/verify_samples.py
```

Sources: nibabel and pydicom test data, the MIT-BIH Arrhythmia Database and the
CHB-MIT Scalp EEG Database (PhysioNet). The tabular and vital-sign files are
synthetic.
