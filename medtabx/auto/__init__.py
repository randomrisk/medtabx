"""MedTabX AutoRecipe: one typed contract, any downstream model.

    from medtabx.auto import build_frame, compare_models, AutoRecipe

    data, builder = build_frame(df, target="poor_outcome",
                                modalities={"demographics": ["age", "sex"],
                                            "labs": ["glucose", "inr"]},
                                images={"dwi": dwi_tokens})
    board = compare_models(data, models=["lr", "cat", "tabpfn@late"])
    auto = AutoRecipe(objective="balanced", budget=60).fit(data)
    print(auto.explain())
    p = auto.predict(builder.transform(new_df, images={"dwi": new_tokens}))
"""

from medtabx.auto.compare import Comparison, compare_models, default_scenarios
from medtabx.auto.frame import (
    FrameBuilder,
    build_frame,
    describe,
    encode_images,
    flat_view,
    token_view,
)
from medtabx.auto.recommend import RULES, AutoRecipe, Objective, prior
from medtabx.auto.zoo import (
    BASES,
    create_model,
    feasible,
    list_models,
    register_base,
)

__all__ = [
    "FrameBuilder", "build_frame", "describe", "encode_images", "flat_view",
    "token_view", "BASES", "create_model", "feasible", "list_models",
    "register_base", "Comparison", "compare_models", "default_scenarios",
    "AutoRecipe", "Objective", "RULES", "prior",
]
