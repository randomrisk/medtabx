"""
Registry system for encoders, interactions, decoders, and transforms.

This provides a centralized registration mechanism for extensibility.
"""
from typing import Dict, Type, Callable, Optional
import warnings


class Registry:
    """Generic registry for components."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class.

        Args:
            name: Optional name for registration. If None, uses class name.

        Example:
            @encoder_registry.register('my_encoder')
            class MyEncoder(BaseEncoder):
                pass
        """
        def decorator(cls: Type) -> Type:
            reg_name = name if name is not None else cls.__name__
            if reg_name in self._registry:
                warnings.warn(
                    f"{reg_name} already registered in {self.name}. "
                    f"Overwriting."
                )
            self._registry[reg_name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        """Get a registered class by name."""
        if name not in self._registry:
            raise KeyError(
                f"{name} not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries
encoder_registry = Registry("Encoder")
interaction_registry = Registry("Interaction")
decoder_registry = Registry("Decoder")
transform_registry = Registry("Transform")
modality_loader_registry = Registry("ModalityLoader")
feature_extractor_registry = Registry("FeatureExtractor")
model_registry = Registry("Model")
tree_model_registry = Registry("TreeModel")
text_embedding_backend_registry = Registry("TextEmbeddingBackend")
