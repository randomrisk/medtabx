#!/usr/bin/env python3
"""MedTabX AutoRecipe in five steps, on a synthetic stroke cohort.

1. describe the table: which columns belong to which modality;
2. build the typed contract (plus frozen image tokens);
3. compare any set of downstream models on it (PyCaret-style leaderboard);
4. let AutoRecipe recommend one, with its reasons;
5. apply the fitted builder + model to new patients.

Runs on CPU with scikit-learn only; LightGBM / XGBoost / CatBoost / TabPFN /
the PyTorch Frame models join automatically when installed.
"""

import numpy as np
import pandas as pd

from medtabx.auto import (AutoRecipe, build_frame, compare_models, describe,
                          list_models, token_view)

rng = np.random.default_rng(0)
n = 200
y = rng.integers(0, 2, n)
df = pd.DataFrame({
    "age": rng.normal(68, 12, n) + 4 * y,
    "sex": rng.choice(["F", "M"], n),
    "nihss": rng.integers(0, 25, n) + 4 * y,
    "atrial_fibrillation": rng.integers(0, 2, n),
    "glucose": rng.normal(6.5, 1.5, n),
    "inr": rng.normal(1.1, 0.2, n),
    "poor_outcome": y,
})
df.loc[rng.random(n) < 0.35, ["glucose", "inr"]] = np.nan     # labs missing
dwi = rng.normal(size=(n, 8, 64)).astype(np.float32)           # frozen tokens
dwi[:, :, :4] += 0.8 * y[:, None, None]
dwi[rng.random(n) < 0.15] = np.nan                              # no DWI

# 1-2. typed contract
modalities = {"demographics": ["age", "sex"],
              "presentation": ["nihss", "atrial_fibrillation"],
              "labs": ["glucose", "inr"]}
train = np.arange(160)
data, builder = build_frame(df.iloc[train], "poor_outcome", modalities,
                            images={"dwi": dwi[train]})
print("contract:", {k: v for k, v in describe(data).items()
                    if k in ("n", "modalities", "value_types",
                             "modality_missing")})
h, _ = token_view(data, d_model=32)
print("token view H [B, N, C]:", tuple(h.shape))
print("available models:", ", ".join(list_models(data)))

# 3. compare models
board = compare_models(data, ["lr", "lr@late", "rf", "rf@late", "hgb",
                              "et@subset"], n_splits=5, log=None)
print(board)

# 4. recommendation
auto = AutoRecipe("balanced", budget=40, n_splits=5, repeats=1).fit(data)
print(auto.explain())

# 5. new patients
test = np.arange(160, n)
new = builder.transform(df.iloc[test], images={"dwi": dwi[test]})
p = auto.predict(new)
print("P(poor outcome) for 5 new patients:", np.round(p[:5], 3))
