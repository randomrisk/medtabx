"""Tests for the MedTabX typed-token transformer."""

from __future__ import annotations

import pytest
import torch

from medtabx.models.typed_token import (
    TypedTokenConfig,
    TypedTokenSchema,
    TypedTokenTransformer,
)


def _schema():
    return TypedTokenSchema(num_modality=["demo", "demo", "labs"],
                            cat_modality=["demo"], cat_cardinality=[3],
                            img_modality=["dwi", "flair"], img_tokens=2,
                            img_dim=16)


def _batch(b=4):
    num = torch.randn(b, 3)
    num[0, 2] = float("nan")
    cat = torch.tensor([[0], [1], [-1], [2]])[:b]
    img = torch.randn(b, 2, 2, 16)
    present = torch.ones(b, 2, dtype=torch.bool)
    present[1, 1] = False
    return num, cat, img, present


def test_forward_shape_and_token_count():
    model = TypedTokenTransformer(_schema())
    assert model.schema.num_tokens == 3 + 1 + 2 * 2
    out = model(*_batch())
    assert out.shape == (4, 1) and torch.isfinite(out).all()


def test_missing_flags_cover_nan_negative_category_and_absent_modality():
    model = TypedTokenTransformer(_schema()).eval()
    _, missing = model.tokens(*_batch())
    assert missing[0, 2]           # NaN numeric
    assert missing[2, 3]           # category -1
    assert missing[1, 6:8].all()   # absent FLAIR: both of its tokens
    assert not missing[3].any()


def test_missing_value_does_not_leak_through():
    # A missing numeric token must not depend on the (NaN-replaced) value.
    model = TypedTokenTransformer(_schema()).eval()
    num, cat, img, present = _batch()
    a = num.clone(); a[0, 2] = float("nan")
    e1, _ = model.tokens(a, cat, img, present)
    e2, _ = model.tokens(a.clone(), cat, img, present)
    assert torch.allclose(e1[0, 2], e2[0, 2])
    img2 = img.clone(); img2[1, 1] += 100.0
    f1, _ = model.tokens(num, cat, img, present)
    f2, _ = model.tokens(num, cat, img2, present)
    assert torch.allclose(f1[1, 6:8], f2[1, 6:8])


def test_output_ignores_absent_modality_content():
    model = TypedTokenTransformer(_schema()).eval()
    num, cat, img, present = _batch()
    img2 = img.clone(); img2[1, 1] = torch.randn(2, 16) * 50
    assert torch.allclose(model(num, cat, img, present)[1],
                          model(num, cat, img2, present)[1], atol=1e-6)


@pytest.mark.parametrize("flag", ["use_type", "use_modality", "use_column",
                                  "use_missing"])
def test_each_ablation_runs(flag):
    cfg = TypedTokenConfig(**{flag: False})
    out = TypedTokenTransformer(_schema(), cfg)(*_batch())
    assert out.shape == (4, 1) and torch.isfinite(out).all()


def test_modality_dropout_only_in_training_and_keeps_one():
    cfg = TypedTokenConfig(modality_dropout=1.0)
    model = TypedTokenTransformer(_schema(), cfg)
    num, cat, img, present = _batch()
    model.train()
    _, missing = model.tokens(num, cat, img, present)
    # p=1 drops every modality, then keep_one restores one per row.
    assert (~missing).any(dim=1).all()
    model.eval()
    _, missing_eval = model.tokens(num, cat, img, present)
    # NaN numeric + missing category + both tokens of the absent FLAIR.
    assert missing_eval.sum() == 4


def test_gradients_flow_to_identity_embeddings():
    model = TypedTokenTransformer(_schema())
    model(*_batch()).sum().backward()
    for emb in (model.type_embedding, model.modality_embedding,
                model.column_embedding, model.missing_embedding):
        assert emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0


def test_gated_experts_head_ignores_absent_modalities_and_has_aux_loss():
    cfg = TypedTokenConfig(head="gated_experts")
    model = TypedTokenTransformer(_schema(), cfg)
    num, cat, img, present = _batch()
    model.train()
    out = model(num, cat, img, present)
    assert out.shape == (4, 1)
    aux = model.auxiliary_loss(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert aux is not None and torch.isfinite(aux)
    model.eval()
    img2 = img.clone(); img2[1, 1] = torch.randn(2, 16) * 100
    assert torch.allclose(model(num, cat, img, present)[1],
                          model(num, cat, img2, present)[1], atol=1e-5)


def test_stacked_head_reduces_to_weighted_expert_logits():
    cfg = TypedTokenConfig(head="stacked")
    model = TypedTokenTransformer(_schema(), cfg).eval()
    num, cat, img, present = _batch()
    n_mod = len(model.schema.modalities)  # demo, labs, dwi, flair
    experts = torch.randn(4, n_mod)
    out = model(num, cat, img, present, expert_logits=experts)
    assert out.shape == (4, 1)
    # Output is a convex combination of the available expert logits.
    lo = experts.min(dim=1).values.unsqueeze(1)
    hi = experts.max(dim=1).values.unsqueeze(1)
    assert ((out >= lo - 1e-5) & (out <= hi + 1e-5)).all()
    # A NaN expert is excluded: changing nothing else, its value is ignored.
    e2 = experts.clone(); e2[0, 0] = float("nan")
    e3 = e2.clone()
    assert torch.allclose(model(num, cat, img, present, expert_logits=e2),
                          model(num, cat, img, present, expert_logits=e3))


def test_stacked_head_requires_expert_logits():
    model = TypedTokenTransformer(_schema(), TypedTokenConfig(head="stacked"))
    with pytest.raises(ValueError, match="expert_logits"):
        model(*_batch())
