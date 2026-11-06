from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from medtabx.core.registry import text_embedding_backend_registry


class TextEmbeddingBackend(ABC):
    """Interface for pluggable text/LLM embedding providers."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@text_embedding_backend_registry.register("deterministic")
class DeterministicTextEmbeddingBackend(TextEmbeddingBackend):
    """Fast local fallback for smoke tests and offline runs."""

    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim

    def _embed_one(self, text: str) -> np.ndarray:
        values = np.zeros(self.embedding_dim, dtype=np.float32)
        if self.embedding_dim >= 1:
            values[0] = len(text)
        if self.embedding_dim >= 2:
            values[1] = text.lower().count("stroke")
        if self.embedding_dim >= 3:
            values[2] = text.lower().count("perfusion")
        if self.embedding_dim >= 4:
            values[3] = text.lower().count("left")
        if self.embedding_dim >= 5:
            values[4] = text.lower().count("right")
        if self.embedding_dim >= 6:
            values[5] = sum(ch.isdigit() for ch in text)
        if self.embedding_dim >= 7:
            values[6] = text.count("\n")
        if self.embedding_dim >= 8:
            values[7] = len(text.split())
        if self.embedding_dim >= 9:
            values[8] = len(set(text.split()))
        if self.embedding_dim >= 10:
            values[9] = sum(ch in "aeiouAEIOU" for ch in text)
        if self.embedding_dim >= 11:
            values[10] = sum(ch.isupper() for ch in text)
        if self.embedding_dim >= 12:
            values[11] = sum(ch.islower() for ch in text)
        if self.embedding_dim >= 13:
            values[12] = text.count(".")
        if self.embedding_dim >= 14:
            values[13] = text.count(",")
        if self.embedding_dim >= 15:
            values[14] = text.count(":")
        if self.embedding_dim >= 16:
            values[15] = text.count("-")
        max_abs = np.abs(values).max()
        if max_abs > 0:
            values = values / max_abs
        return values

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed_one(str(text)) for text in texts], axis=0)


@text_embedding_backend_registry.register("transformers")
class TransformersTextEmbeddingBackend(TextEmbeddingBackend):
    """Optional Hugging Face encoder with mean pooling."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(target_device)
        model.eval()

        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * attention_mask
            denom = attention_mask.sum(dim=1).clamp_min(1)
            pooled = masked.sum(dim=1) / denom
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


class CallableTextEmbeddingBackend(TextEmbeddingBackend):
    """Wrap a custom embedding callable or API client."""

    def __init__(self, fn: Callable[[Sequence[str]], Any]):
        self.fn = fn

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        values = self.fn(texts)
        return np.asarray(values, dtype=np.float32)


def get_text_embedding_backend(
    backend: str | TextEmbeddingBackend | Callable[[Sequence[str]], Any] = "deterministic",
    **backend_kwargs,
) -> TextEmbeddingBackend:
    if isinstance(backend, TextEmbeddingBackend):
        return backend
    if callable(backend) and not isinstance(backend, str):
        return CallableTextEmbeddingBackend(backend)
    backend_cls = text_embedding_backend_registry.get(str(backend))
    return backend_cls(**backend_kwargs)


def embed_texts(
    texts: Sequence[str],
    backend: str | TextEmbeddingBackend | Callable[[Sequence[str]], Any] = "deterministic",
    **backend_kwargs,
) -> np.ndarray:
    return get_text_embedding_backend(backend, **backend_kwargs).embed(texts)
