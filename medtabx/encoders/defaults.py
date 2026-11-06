"""
Default encoder selection.

Two encoder hierarchies coexist in MedTabX and this module serves both, which
is why it exposes two entry points rather than one:

* The **tabular path** (:class:`StypeWiseFeatureEncoder` +
  :class:`StypeEncoder`) consumes a whole :class:`TensorFrame` and is what the
  tabular models (FT-Transformer, ResNet, MLP, ...) are built from.
  :func:`default_tabular_stype_encoder_dict` covers every stype that has a
  :class:`StypeEncoder` implementation.

* The **multimodal path** (:class:`~medtabx.models.MultimodalBaseline` +
  :class:`BaseEncoder`) maps each stype's raw feature tensor to
  ``[batch_size, num_cols, embedding_dim]``. Those encoders need shape
  arguments -- channel counts, sequence lengths, embedding widths -- that are
  properties of the materialized data, not of the task.
  :func:`build_stype_encoder_dict` reads them off the
  :class:`TensorFrame` so callers do not have to hand-wire them.

Both entry points return **fresh module instances** on every call.
:class:`StypeWiseFeatureEncoder` mutates the encoders it is given (it assigns
``stype``, ``out_channels`` and ``stats_list``), so sharing one instance
between two models would silently couple them.
"""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor
from torch.nn import Module

import medtabx
from medtabx.core.stype import stype
from medtabx.data.stats import StatType

from .base import BaseEncoder
from .categorical import CategoricalEncoder
from .ecg import ECGEncoder
from .eeg import EEGEncoder
from .fmri import FMRIEncoder
from .imaging3d import Imaging3DEncoder
from .lab_value import LabValueEncoder
from .mask3d import Mask3DEncoder
from .medical_code import MedicalCodeEncoder
from .medication import MedicationEncoder
from .numeric import NumericEncoder
from .stype_encoder import (
    EmbeddingEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    MultiCategoricalEmbeddingEncoder,
    StypeEncoder,
    TimestampEncoder,
)
from .text import TextEmbeddedEncoder
from .vital_sign import VitalSignEncoder

__all__ = [
    "DEFAULT_TABULAR_ENCODER_CLS",
    "DEFAULT_MEDICAL_ENCODER_CLS",
    "default_tabular_stype_encoder_dict",
    "infer_encoder_kwargs",
    "build_stype_encoder_dict",
]

#: Default :class:`StypeEncoder` class per stype, for the tabular path.
DEFAULT_TABULAR_ENCODER_CLS: dict[stype, type[StypeEncoder]] = {
    stype.numerical: LinearEncoder,
    stype.categorical: EmbeddingEncoder,
    stype.multicategorical: MultiCategoricalEmbeddingEncoder,
    stype.embedding: LinearEmbeddingEncoder,
    stype.timestamp: TimestampEncoder,
}

#: Default :class:`BaseEncoder` class per stype, for the multimodal path.
DEFAULT_MEDICAL_ENCODER_CLS: dict[stype, type[BaseEncoder]] = {
    stype.numerical: NumericEncoder,
    stype.categorical: CategoricalEncoder,
    stype.embedding: TextEmbeddedEncoder,
    stype.image_3d: Imaging3DEncoder,
    stype.mask_3d: Mask3DEncoder,
    stype.timeseries_4d: FMRIEncoder,
    stype.fmri: FMRIEncoder,
    stype.eeg: EEGEncoder,
    stype.ecg: ECGEncoder,
    stype.vital_sign: VitalSignEncoder,
    stype.lab_value: LabValueEncoder,
    stype.medical_code: MedicalCodeEncoder,
    stype.medication: MedicationEncoder,
}


def default_tabular_stype_encoder_dict(
    stypes: list[stype] | None = None,
    overrides: Mapping[stype, StypeEncoder] | None = None,
) -> dict[stype, StypeEncoder]:
    r"""Default :class:`StypeEncoder` per stype for the tabular path.

    Tabular models call this when no ``stype_encoder_dict`` is supplied.
    Covering every encodable stype -- rather than only numerical and
    categorical -- means a frame that carries report embeddings, timestamps or
    multicategorical codes trains without the caller having to discover the
    omission from a failed forward pass.

    Args:
        stypes (list, optional): Restrict the result to these stypes, which
            should be the keys of the frame's ``col_names_dict``. Passing the
            frame's stypes avoids constructing encoders the model will never
            use. An stype with no default raises. (default: :obj:`None`)
        overrides (dict, optional): Encoder instances to use in place of the
            defaults. (default: :obj:`None`)

    Example:
        >>> from medtabx.encoders import default_tabular_stype_encoder_dict
        >>> enc = default_tabular_stype_encoder_dict(
        ...     stypes=list(dataset.tensor_frame.col_names_dict))
    """
    overrides = dict(overrides or {})

    if stypes is None:
        selected = list(DEFAULT_TABULAR_ENCODER_CLS)
    else:
        selected = []
        unsupported = []
        for st in stypes:
            parent = st.parent
            if parent in overrides or parent in DEFAULT_TABULAR_ENCODER_CLS:
                if parent not in selected:
                    selected.append(parent)
            else:
                unsupported.append(st)
        if unsupported:
            raise ValueError(
                f"No default tabular encoder for "
                f"{sorted(s.value for s in unsupported)}. The tabular path "
                f"supports "
                f"{sorted(s.value for s in DEFAULT_TABULAR_ENCODER_CLS)}; "
                f"raw medical modalities go through MultimodalBaseline and "
                f"build_stype_encoder_dict instead. Alternatively pass an "
                f"encoder for them via stype_encoder_dict.")

    encoder_dict: dict[stype, StypeEncoder] = {}
    for st in selected:
        if st in overrides:
            encoder_dict[st] = overrides[st]
        else:
            encoder_dict[st] = DEFAULT_TABULAR_ENCODER_CLS[st]()
    return encoder_dict


def _num_categories(col_stats: Mapping[str, Any] | None,
                    col_names: list[str] | None) -> int:
    r"""Widest category vocabulary over the given columns.

    :class:`CategoricalEncoder` holds a single embedding table shared by every
    categorical column, so it has to be sized for the widest one.
    """
    if not col_stats or not col_names:
        raise ValueError(
            "Encoding a categorical stype needs col_stats to size the "
            "embedding table. Pass the dataset's col_stats, or supply a "
            "CategoricalEncoder through 'overrides'.")

    sizes = []
    for col in col_names:
        stats = col_stats.get(col)
        if stats and StatType.COUNT in stats:
            sizes.append(len(stats[StatType.COUNT][0]))
    if not sizes:
        raise ValueError(
            f"None of the categorical columns {col_names} carry a "
            f"StatType.COUNT statistic, so the embedding table cannot be "
            f"sized.")
    return max(sizes)


def infer_encoder_kwargs(
    st: stype,
    feat: Tensor,
    col_stats: Mapping[str, Any] | None = None,
    col_names: list[str] | None = None,
) -> dict[str, Any]:
    r"""Shape arguments for an stype's default multimodal encoder.

    Materialization stacks columns on axis 1, so every feature tensor has the
    layout ``[batch_size, num_cols, *per_column_shape]``; the per-column shape
    is what the encoders need.

    Args:
        st (stype): The stype to configure. Should be a parent stype, i.e. a
            key of a materialized ``feat_dict``.
        feat (Tensor): That stype's feature tensor.
        col_stats (dict, optional): Dataset column statistics, required for
            :obj:`categorical`. (default: :obj:`None`)
        col_names (list, optional): The stype's column names, required for
            :obj:`categorical`. (default: :obj:`None`)

    Returns:
        dict: Keyword arguments, excluding ``embedding_dim``.
    """
    if not isinstance(feat, Tensor):
        raise TypeError(
            f"Cannot infer encoder arguments for stype '{st.value}' from a "
            f"{type(feat).__name__}. The multimodal encoders take dense "
            f"tensors; supply an encoder for this stype through "
            f"'overrides'.")

    if st == stype.numerical:
        return {"num_cols": feat.shape[1]}
    if st == stype.categorical:
        return {"num_categories": _num_categories(col_stats, col_names)}
    if st == stype.embedding:
        return {"input_dim": feat.shape[-1]}
    if st == stype.image_3d:
        return {"in_channels": feat.shape[2]}
    if st == stype.mask_3d:
        return {"num_classes": feat.shape[2]}
    if st in (stype.timeseries_4d, stype.fmri):
        return {"num_rois": feat.shape[2], "seq_len": feat.shape[-1]}
    if st in (stype.eeg, stype.ecg):
        return {"num_channels": feat.shape[2], "seq_len": feat.shape[-1]}
    if st == stype.vital_sign:
        return {"seq_len": feat.shape[-1]}
    # lab_value, medical_code and medication take only their defaults.
    return {}


def build_stype_encoder_dict(
    tensor_frame: medtabx.data.TensorFrame,
    embedding_dim: int,
    col_stats: Mapping[str, Any] | None = None,
    overrides: Mapping[stype, Module] | None = None,
    include: list[stype] | None = None,
    exclude: list[stype] | None = None,
    strict: bool = True,
) -> dict[stype, Module]:
    r"""Build a multimodal encoder per stype present in a materialized frame.

    Every shape argument is read off the frame, so adding a modality to the
    dataset is enough to get it encoded -- there is no second place to update.

    .. code-block:: python

        from medtabx.encoders import build_stype_encoder_dict
        from medtabx.models import MultimodalBaseline

        dataset.materialize()
        model = MultimodalBaseline(
            stype_encoder_dict=build_stype_encoder_dict(
                dataset.tensor_frame, embedding_dim=64,
                col_stats=dataset.col_stats),
            decoder=ClassificationDecoder(embedding_dim=64, num_classes=2),
            embedding_dim=64,
        )

    Args:
        tensor_frame (TensorFrame): A materialized frame.
        embedding_dim (int): Common output width for every encoder.
        col_stats (dict, optional): Dataset column statistics, required when
            the frame holds categorical columns. (default: :obj:`None`)
        overrides (dict, optional): Encoder instances to use in place of the
            defaults, keyed by stype. An stype mapped here is used as given
            and no arguments are inferred for it. (default: :obj:`None`)
        include (list, optional): Encode only these stypes. This is how a
            single-modality arm of an ablation is expressed: the dataset stays
            the same and only the encoder set changes, so the comparison is
            not confounded by a different data path.
            (default: :obj:`None`)
        exclude (list, optional): Skip these stypes. Applied after
            :obj:`include`. Useful for dropping identifier-like columns and
            for modality-dropout experiments. (default: :obj:`None`)
        strict (bool): Raise when the frame carries an stype that has no
            default multimodal encoder. When :obj:`False` that stype is
            skipped, which drops those columns from the model.
            (default: :obj:`True`)

    Returns:
        dict: Maps stype to a ready encoder module.
    """
    overrides = dict(overrides or {})
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or ())

    if include_set is not None:
        absent = include_set - set(tensor_frame.feat_dict)
        if absent:
            raise ValueError(
                f"Cannot include {sorted(s.value for s in absent)}: the "
                f"frame only holds "
                f"{sorted(s.value for s in tensor_frame.feat_dict)}.")

    encoder_dict: dict[stype, Module] = {}
    unsupported: list[stype] = []

    for st, feat in tensor_frame.feat_dict.items():
        if include_set is not None and st not in include_set:
            continue
        if st in exclude_set:
            continue
        if st in overrides:
            encoder_dict[st] = overrides[st]
            continue

        encoder_cls = DEFAULT_MEDICAL_ENCODER_CLS.get(st)
        if encoder_cls is None:
            unsupported.append(st)
            continue

        kwargs = infer_encoder_kwargs(
            st, feat, col_stats=col_stats,
            col_names=tensor_frame.col_names_dict.get(st))
        encoder_dict[st] = encoder_cls(embedding_dim=embedding_dim, **kwargs)

    if unsupported and strict:
        raise ValueError(
            f"No default multimodal encoder for "
            f"{sorted(s.value for s in unsupported)}. Supply one through "
            f"'overrides', or pass strict=False to drop those columns.")

    if not encoder_dict:
        raise ValueError(
            "No encoders were built. Check that the dataset is materialized "
            "and that 'include'/'exclude' did not filter out every stype in "
            "the frame.")

    return encoder_dict
