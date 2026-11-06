"""MedTabX typed-token interaction model.

Every input becomes a token of width ``d``:

    E = E_value + E_type + E_modality + E_column + E_missing

- ``E_value``    what was measured: a per-column affine map of a numeric value,
                 an embedding lookup for a category, or a projection of a
                 frozen foundation-encoder image token.
- ``E_type``     the semantic type of the column (numerical / categorical /
                 image) -- MedTabX's stype.
- ``E_modality`` where the value came from (e.g. demographics, labs, DWI,
                 FLAIR, lesion QFV), shared by every column of that source.
- ``E_column``   which column it is (for image tokens: modality x position).
- ``E_missing``  whether the value is observed. A missing value contributes no
                 value signal; its token carries the missing state instead, so
                 the model can tell "absent" from "zero".

All tokens, a CLS token, and nothing else form one sequence for a standard
transformer, so heterogeneous clinical, derived and imaging inputs interact
directly. Training can drop whole modalities at random
(``modality_dropout``), which is what teaches ``E_missing`` its meaning and
makes the model robust to modalities that are unavailable at test time.

Switching embeddings off reproduces familiar baselines at matched capacity:
column-only identity is the FT-Transformer feature tokenizer, and no identity
at all is a plain transformer over value tokens. Use these switches for
ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["TypedTokenConfig", "TypedTokenSchema", "TypedTokenTransformer"]

TYPE_NUMERICAL, TYPE_CATEGORICAL, TYPE_IMAGE = 0, 1, 2


@dataclass
class TypedTokenSchema:
    r"""Column layout of a typed multimodal dataset.

    Attributes:
        num_modality: Modality name for each numerical column.
        cat_modality: Modality name for each categorical column.
        cat_cardinality: Number of categories for each categorical column.
        img_modality: Name of each image modality.
        img_tokens: Tokens per image modality (``K``).
        img_dim: Width of an image token from the frozen encoder.
    """

    num_modality: list[str] = field(default_factory=list)
    cat_modality: list[str] = field(default_factory=list)
    cat_cardinality: list[int] = field(default_factory=list)
    img_modality: list[str] = field(default_factory=list)
    img_tokens: int = 1
    img_dim: int = 320

    @property
    def modalities(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in (self.num_modality + self.cat_modality +
                     self.img_modality):
            seen.setdefault(name, None)
        return list(seen)

    @property
    def num_tokens(self) -> int:
        return (len(self.num_modality) + len(self.cat_modality) +
                len(self.img_modality) * self.img_tokens)


@dataclass
class TypedTokenConfig:
    r"""Hyperparameters and embedding switches.

    The four ``use_*`` switches turn single embeddings on or off for
    ablation studies.
    """

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    out_dim: int = 1
    use_type: bool = True
    use_modality: bool = True
    use_column: bool = True
    use_missing: bool = True
    #: Probability of dropping each whole modality during training.
    modality_dropout: float = 0.0
    #: Keep at least one modality when dropping.
    keep_one: bool = True
    #: Add a linear ("wide") readout over the raw inputs to the transformer
    #: output (wide & deep). Small cohorts are where transformers alone lose
    #: to regularized linear models; the wide path keeps that linear signal
    #: while the typed transformer models interactions. Missing inputs
    #: contribute nothing to the wide path.
    wide: bool = False
    #: ``"stacked"``: typed stacking -- combine externally fitted
    #: per-modality expert logits (passed as ``expert_logits``) with the same
    #: masked softmax gate. With a uniform gate this is exactly late fusion.
    #: ``"cls"``: predict from the CLS token (plus the wide path).
    #: ``"gated_experts"``: typed gated fusion -- one linear expert per
    #: modality, combined by a softmax gate computed from the typed
    #: transformer and renormalized over the modalities actually present.
    #: This generalizes late fusion (a fixed equal average over present
    #: modalities) with a patient-specific, missingness-aware weighting.
    head: str = "cls"
    #: Weight of the per-expert auxiliary losses (each expert is also trained
    #: to predict alone, so it stays usable when others are missing).
    aux_weight: float = 1.0
    #: Weight of the modality-alignment loss (0 = off, the default, so every
    #: earlier result is unchanged). Alignment penalizes disagreement between
    #: the per-modality experts of the *same* patient, which is the explicit
    #: cross-modal constraint that task loss alone does not impose
    #: (Yang et al., TPAMI 2026).
    align_weight: float = 0.0
    #: ``"js"``: Jensen-Shannon divergence between expert predictive
    #: distributions (soft, label-free, usable on unlabeled rows).
    #: ``"rep"``: cosine alignment of the per-modality token representations.
    #: ``"hybrid"``: both.
    align: str = "js"
    #: How the alignment weight changes over training (Yang et al., Sec. 3.3):
    #: ``"const"``; ``"decay"`` = their heuristic alpha_t = 1 - exp(-1/t),
    #: alignment first, fusion later; ``"stepwise"`` = alignment for the first
    #: half of the epochs only.
    align_schedule: str = "const"
    #: ``"add"``: task loss + alpha * alignment. ``"convex"``: their Eq. 12,
    #: (1 - alpha) * task + alpha * alignment.
    align_integration: str = "add"
    #: Weight of the entropy term on unlabeled rows (their Eq. 17), used only
    #: when a semi-supervised set is passed to the training method.
    ssl_entropy: float = 0.0


class TypedTokenTransformer(nn.Module):
    r"""Typed multimodal token transformer.

    Forward inputs (any group may be absent for a dataset without it):

    - ``num``  ``[B, Cn]`` float, standardized; ``NaN`` marks missing.
    - ``cat``  ``[B, Cc]`` long, ``-1`` marks missing.
    - ``img``  ``[B, M, K, D]`` float image tokens.
    - ``img_present`` ``[B, M]`` bool.

    Returns ``[B, out_dim]`` logits (or regression outputs).
    """

    def __init__(self, schema: TypedTokenSchema,
                 config: TypedTokenConfig | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.config = config = config or TypedTokenConfig()
        d = config.d_model

        modalities = schema.modalities
        self.modality_index = {m: i for i, m in enumerate(modalities)}
        cn, cc, m = (len(schema.num_modality), len(schema.cat_modality),
                     len(schema.img_modality))
        k = schema.img_tokens

        # E_value
        # Per-column scale only. A per-column bias would smuggle column
        # identity into the value and undermine the E_column ablation.
        self.num_weight = nn.Parameter(torch.randn(cn, d) * 0.1) if cn else None
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(card, d) for card in schema.cat_cardinality)
        # Plain linear projection. Image tokens should arrive standardized
        # per dimension (as the harness does); a per-token LayerNorm here
        # would erase each token's overall level, which can carry signal.
        self.img_proj = nn.Linear(schema.img_dim, d) if m else None

        # Identity embeddings
        self.type_embedding = nn.Embedding(3, d)
        self.modality_embedding = nn.Embedding(max(1, len(modalities)), d)
        self.column_embedding = nn.Embedding(max(1, cn + cc + m * k), d)
        self.missing_embedding = nn.Embedding(2, d)

        # Static per-token index tables, in sequence order num | cat | img.
        types = ([TYPE_NUMERICAL] * cn + [TYPE_CATEGORICAL] * cc +
                 [TYPE_IMAGE] * (m * k))
        mods = ([self.modality_index[x] for x in schema.num_modality] +
                [self.modality_index[x] for x in schema.cat_modality] +
                [self.modality_index[x] for x in schema.img_modality
                 for _ in range(k)])
        # Modality id of each token, used for modality dropout.
        self.register_buffer("token_type", torch.tensor(types, dtype=torch.long))
        self.register_buffer("token_modality", torch.tensor(mods, dtype=torch.long))
        self.register_buffer("token_column",
                             torch.arange(len(types), dtype=torch.long))

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=2 * d,
            dropout=config.dropout, batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, config.n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.out_dim))
        # Prediction-level alignment compares per-modality predictions, so
        # those predictions must lie on the path that produces the output --
        # otherwise the experts get no task gradient and the loss has the
        # trivial solution "every expert predicts the base rate", where the
        # divergence is 0 and nothing is aligned. In Yang et al. the branches
        # are the fused linear classifier split per modality; the equivalent
        # here is the gated-experts head, whose output is a weighted sum of
        # the experts. Representation alignment has no such problem: the
        # token states are always on the path.
        if config.align_weight > 0 and config.align in ("js", "hybrid") \
                and config.head != "gated_experts":
            raise ValueError(
                f"align={config.align!r} needs head='gated_experts' so the "
                "per-modality predictions are part of the output; with "
                f"head={config.head!r} the alignment term is vacuous. Use "
                "align='rep' to align representations instead.")
        # Per-modality experts: the head that fuses them, and any alignment
        # loss, both need them (alignment compares the modalities of one
        # patient, so it needs a prediction per modality).
        if config.head == "gated_experts" or config.align_weight > 0:
            o = config.out_dim
            n_mod = len(modalities)
            self.n_modalities = n_mod
            self.expert_num = nn.Linear(cn, n_mod * o) if cn else None
            self.expert_num_missing = (nn.Linear(cn, n_mod * o, bias=False)
                                       if cn else None)
            self.expert_cat = nn.ModuleList(
                nn.Embedding(card, n_mod * o) for card in schema.cat_cardinality)
            self.expert_img = nn.ModuleList(
                nn.Linear(schema.img_dim, o) for _ in range(m))
            self.expert_bias = nn.Parameter(torch.zeros(n_mod, o))
            self.gate = nn.Linear(d, n_mod)
            # Which modality each numeric / categorical column feeds.
            num_mod_idx = [self.modality_index[x] for x in schema.num_modality]
            cat_mod_idx = [self.modality_index[x] for x in schema.cat_modality]
            img_mod_idx = [self.modality_index[x] for x in schema.img_modality]
            self.register_buffer("num_col_modality",
                                 torch.tensor(num_mod_idx, dtype=torch.long))
            self.register_buffer("cat_col_modality",
                                 torch.tensor(cat_mod_idx, dtype=torch.long))
            self.register_buffer("img_col_modality",
                                 torch.tensor(img_mod_idx, dtype=torch.long))
            self.aux_loss = None
            if config.head != "gated_experts":
                # Used only by the alignment loss; the prediction path is
                # unchanged, so a run with align_weight=0 is bit-identical.
                del self.gate
        if config.head == "stacked":
            self.gate = nn.Linear(d, len(modalities))
            # Calibration of the combined logit (starts as identity).
            self.stack_scale = nn.Parameter(torch.ones(1))
            self.stack_bias = nn.Parameter(torch.zeros(1))
        if config.wide:
            o = config.out_dim
            self.wide_num = nn.Linear(cn, o) if cn else None
            self.wide_num_missing = nn.Linear(cn, o, bias=False) if cn else None
            self.wide_cat = nn.ModuleList(
                nn.Embedding(card, o) for card in schema.cat_cardinality)
            self.wide_img = nn.ModuleList(
                nn.Linear(schema.img_dim, o) for _ in range(m))
        self._reset()

    def _reset(self) -> None:
        for emb in (self.type_embedding, self.modality_embedding,
                    self.column_embedding, self.missing_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    # Token construction ####################################################

    def _values(self, num, cat, img, img_present):
        """Value embeddings ``[B, T, d]`` and missing flags ``[B, T]``."""
        parts, missing = [], []
        if num is not None and num.shape[1]:
            miss = torch.isnan(num)
            x = torch.nan_to_num(num, nan=0.0)
            v = x.unsqueeze(-1) * self.num_weight
            parts.append(v * (~miss).unsqueeze(-1))
            missing.append(miss)
        if cat is not None and cat.shape[1]:
            miss = cat < 0
            v = torch.stack([
                emb(cat[:, j].clamp_min(0))
                for j, emb in enumerate(self.cat_embeddings)
            ], dim=1)
            parts.append(v * (~miss).unsqueeze(-1))
            missing.append(miss)
        if img is not None and img.shape[1]:
            b, m, k, _ = img.shape
            v = self.img_proj(img.float()).reshape(b, m * k, -1)
            miss = (~img_present).repeat_interleave(k, dim=1)
            parts.append(v * (~miss).unsqueeze(-1))
            missing.append(miss)
        return torch.cat(parts, dim=1), torch.cat(missing, dim=1)

    def _modality_dropout(self, missing: Tensor) -> Tensor:
        p = self.config.modality_dropout
        if not self.training or p <= 0:
            return missing
        b = missing.shape[0]
        n_mod = int(self.token_modality.max()) + 1
        drop = torch.rand(b, n_mod, device=missing.device) < p
        if self.config.keep_one:
            # Never leave a row with nothing observed: if dropping would do
            # that, restore one modality the subject actually has.
            observed = torch.zeros(b, n_mod, device=missing.device)
            observed.scatter_add_(
                1, self.token_modality.expand(b, -1), (~missing).float())
            observed = observed > 0
            survives = (observed & ~drop).any(dim=1)
            rows = (~survives) & observed.any(dim=1)
            if rows.any():
                keep = torch.multinomial(observed[rows].float(), 1).squeeze(1)
                drop[rows.nonzero(as_tuple=True)[0], keep] = False
        return missing | drop[:, self.token_modality]

    def tokens(self, num=None, cat=None, img=None,
               img_present=None) -> tuple[Tensor, Tensor]:
        """Token sequence ``[B, T, d]`` and the missing flags ``[B, T]``."""
        values, missing = self._values(num, cat, img, img_present)
        missing = self._modality_dropout(missing)
        # Dropped tokens lose their value, like truly missing ones.
        values = values * (~missing).unsqueeze(-1)
        cfg = self.config
        e = values
        if cfg.use_type:
            e = e + self.type_embedding(self.token_type)
        if cfg.use_modality:
            e = e + self.modality_embedding(self.token_modality)
        if cfg.use_column:
            e = e + self.column_embedding(self.token_column)
        if cfg.use_missing:
            e = e + self.missing_embedding(missing.long())
        return e, missing

    def _wide(self, num, cat, img, img_present, missing) -> Tensor:
        """Linear readout over observed raw inputs (``missing`` already
        includes modality dropout, so dropped inputs are excluded here too).
        """
        out = 0.0
        t = 0
        if num is not None and num.shape[1]:
            cn = num.shape[1]
            miss = missing[:, t:t + cn]
            x = torch.nan_to_num(num, nan=0.0) * (~miss)
            out = out + self.wide_num(x) + self.wide_num_missing(miss.float())
            t += cn
        if cat is not None and cat.shape[1]:
            cc = cat.shape[1]
            miss = missing[:, t:t + cc]
            for j, emb in enumerate(self.wide_cat):
                out = out + emb(cat[:, j].clamp_min(0)) * (~miss[:, j:j + 1])
            t += cc
        if img is not None and img.shape[1]:
            k = img.shape[2]
            for j, lin in enumerate(self.wide_img):
                present = ~missing[:, t + j * k]
                out = out + lin(img[:, j].float().mean(1)) * present.unsqueeze(1)
        return out

    def _experts(self, num, cat, img, missing):
        """Per-modality expert logits ``[B, M, o]`` and presence ``[B, M]``.

        Each expert only sees its own modality's observed inputs: for numeric
        columns a masked linear map (column j contributes only to its own
        modality's expert), for categories an embedding, for images a linear
        map of the mean token.
        """
        b = missing.shape[0]
        n_mod, o = self.n_modalities, self.config.out_dim
        logits = self.expert_bias.expand(b, -1, -1).clone()
        observed = torch.zeros(b, n_mod, device=missing.device)
        t = 0
        if num is not None and num.shape[1]:
            cn = num.shape[1]
            miss = missing[:, t:t + cn]
            x = torch.nan_to_num(num, nan=0.0) * (~miss)
            # Route each column to its modality by masking the weight.
            own = F.one_hot(self.num_col_modality, n_mod).float()  # [cn, M]
            w = self.expert_num.weight.view(n_mod, o, cn)
            wm = self.expert_num_missing.weight.view(n_mod, o, cn)
            mask = own.t().unsqueeze(1)  # [M, 1, cn]
            logits = logits + torch.einsum("bc,moc->bmo", x, w * mask)
            logits = logits + torch.einsum("bc,moc->bmo", miss.float(), wm * mask)
            observed.index_add_(1, self.num_col_modality, (~miss).float())
            t += cn
        if cat is not None and cat.shape[1]:
            cc = cat.shape[1]
            miss = missing[:, t:t + cc]
            for j, emb in enumerate(self.expert_cat):
                mj = int(self.cat_col_modality[j])
                v = emb(cat[:, j].clamp_min(0)).view(b, n_mod, o)[:, mj]
                logits[:, mj] = logits[:, mj] + v * (~miss[:, j:j + 1])
            observed.index_add_(1, self.cat_col_modality, (~miss).float())
            t += cc
        if img is not None and img.shape[1]:
            k = img.shape[2]
            for j, lin in enumerate(self.expert_img):
                present = ~missing[:, t + j * k]
                mj = int(self.img_col_modality[j])
                logits[:, mj] = logits[:, mj] + (
                    lin(img[:, j].float().mean(1)) * present.unsqueeze(1))
                observed[:, mj] += present.float()
        return logits, observed > 0

    def modality_present(self, missing: Tensor) -> Tensor:
        """``[B, M]``: whether any token of each modality is observed."""
        b = missing.shape[0]
        n_mod = int(self.token_modality.max()) + 1
        observed = torch.zeros(b, n_mod, device=missing.device)
        observed.scatter_add_(1, self.token_modality.expand(b, -1),
                              (~missing).float())
        return observed > 0

    def forward(self, num=None, cat=None, img=None, img_present=None,
                expert_logits: Tensor | None = None) -> Tensor:
        e, missing = self.tokens(num, cat, img, img_present)
        b = e.shape[0]
        x = torch.cat([self.cls.expand(b, -1, -1), e], dim=1)
        # Without a missing embedding the model cannot tell an absent token
        # from a zero one, so absent tokens are masked out of attention
        # instead; with it, they stay visible and carry the missing state.
        if self.config.use_missing:
            key_padding = None
        else:
            key_padding = torch.cat(
                [torch.zeros(b, 1, dtype=torch.bool, device=e.device),
                 missing], dim=1)
        h = self.backbone(x, src_key_padding_mask=key_padding)
        if self.training and self.config.align_weight > 0:
            # Alignment needs per-modality views; the experts give them for
            # any head, and the token states give the representation view.
            self._align = (*self._experts(num, cat, img, missing),
                           self._modality_states(h[:, 1:], missing))
        if self.config.head == "stacked":
            if expert_logits is None:
                raise ValueError("head='stacked' needs expert_logits [B, M].")
            # A modality counts as present only if its tokens are observed
            # (after modality dropout) and its expert produced a prediction.
            present = self.modality_present(missing) & ~torch.isnan(
                expert_logits)
            ex = torch.nan_to_num(expert_logits, nan=0.0)
            gate = self.gate(h[:, 0]).masked_fill(~present, float("-inf"))
            none = ~present.any(dim=1)
            gate = gate.masked_fill(none.unsqueeze(1), 0.0)
            weights = torch.softmax(gate, dim=1)
            combined = (weights * ex).sum(dim=1, keepdim=True)
            return self.stack_scale * combined + self.stack_bias
        if self.config.head == "gated_experts":
            experts, present = self._experts(num, cat, img, missing)
            gate = self.gate(h[:, 0]).masked_fill(~present, float("-inf"))
            # A row with nothing observed would give all -inf; fall back to
            # uniform weights there (its experts are all bias-only).
            none = ~present.any(dim=1)
            if none.any():
                gate = gate.masked_fill(none.unsqueeze(1), 0.0)
            weights = torch.softmax(gate, dim=1)
            out = (weights.unsqueeze(-1) * experts).sum(dim=1)
            self.aux_loss = None
            if self.training:
                self._aux = (experts, present)
            return out
        out = self.head(h[:, 0])
        if self.config.wide:
            out = out + self._wide(num, cat, img, img_present, missing)
        return out

    def _modality_states(self, h: Tensor, missing: Tensor) -> Tensor:
        """Mean token state per modality ``[B, M, d]`` over observed tokens."""
        b, _, d = h.shape
        obs = (~missing).float().unsqueeze(-1)
        idx = self.token_modality
        total = torch.zeros(b, self.n_modalities, d, device=h.device,
                            dtype=h.dtype)
        count = torch.zeros(b, self.n_modalities, 1, device=h.device,
                            dtype=h.dtype)
        total.index_add_(1, idx, h * obs)
        count.index_add_(1, idx, obs)
        return total / count.clamp_min(1.0)

    def alignment_loss(self) -> Tensor | None:
        """Disagreement between the modalities of the same patient.

        ``js``: Jensen-Shannon divergence between the expert predictive
        distributions, averaged over the pairs of modalities that are both
        observed. ``rep``: 1 - cosine similarity between the per-modality
        token states. No labels are used, so it also applies to unlabeled
        rows. Call after a training forward pass.
        """
        if not hasattr(self, "_align"):
            return None
        experts, present, states = self._align
        del self._align
        pairs = torch.triu_indices(self.n_modalities, self.n_modalities,
                                   offset=1, device=experts.device)
        i, j = pairs[0], pairs[1]
        if i.numel() == 0:
            return None
        both = (present[:, i] & present[:, j]).float()
        if both.sum() == 0:
            return None
        terms = []
        if self.config.align in ("js", "hybrid"):
            p = torch.sigmoid(experts[..., 0])
            p = torch.stack([1 - p, p], dim=-1).clamp_min(1e-6)
            pi, pj = p[:, i], p[:, j]
            m = 0.5 * (pi + pj)
            js = 0.5 * ((pi * (pi.log() - m.log())).sum(-1)
                        + (pj * (pj.log() - m.log())).sum(-1))
            terms.append((js * both).sum() / both.sum())
        if self.config.align in ("rep", "hybrid"):
            z = F.normalize(states, dim=-1)
            cos = (z[:, i] * z[:, j]).sum(-1)
            terms.append(((1 - cos) * both).sum() / both.sum())
        return sum(terms)

    def auxiliary_loss(self, target: Tensor) -> Tensor | None:
        """Per-expert BCE on observed modalities (gated_experts head only).

        Call after a training forward pass. Trains each expert to predict on
        its own, like the separate models of late fusion.
        """
        if self.config.head != "gated_experts" or not hasattr(self, "_aux"):
            return None
        experts, present = self._aux
        del self._aux
        t = target.view(-1, 1).expand(-1, experts.shape[1])
        loss = F.binary_cross_entropy_with_logits(
            experts[..., 0], t, reduction="none")
        denom = present.float().sum().clamp_min(1.0)
        return (loss * present.float()).sum() / denom
